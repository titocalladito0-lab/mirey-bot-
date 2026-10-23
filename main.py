from __future__ import annotations

import asyncio
import io
import os
import logging
import time
from collections import defaultdict

from agent import AgentRequest, MireyAgent
from config import settings
from memory import MemoryStore, context_key_for_dm, context_key_for_guild
from metrics import MetricsCollector, MessageSnapshot
from observability import run_observability_server
from personas import PeopleStore
from storage import Database

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO) if hasattr(settings, 'log_level') else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mirey")
log.setLevel(logging.DEBUG)


def _import_discord():
    try:
        import discord
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install runtime dependencies with `pip install -r requirements.txt`.") from exc
    return discord


def build_client():
    discord = _import_discord()
    intents = discord.Intents.default()
    intents.guilds = True
    intents.message_content = True
    try:
        intents.members = True
    except AttributeError:
        pass

    client = discord.Client(intents=intents)
    db = Database(settings.db_path)
    db.init()
    memory = MemoryStore(db, settings.context_token_budget)
    people = PeopleStore.load(settings.people_path)
    mirey = MireyAgent(memory=memory, people=people, db=db, config=settings)
    metrics = MetricsCollector()
    server_queues = defaultdict(asyncio.Queue)
    active_workers: set[str] = set()
    worker_lock = asyncio.Lock()

    @client.event
    async def on_ready():
        guild_names = ", ".join(guild.name for guild in client.guilds)
        log.info("Connected as %s. Guilds: %s", client.user, guild_names or "none")
        log.info("Observability dashboard at http://127.0.0.1:8765")

    async def handle_message(message, is_dm: bool) -> None:
        author_id = str(message.author.id)

        context_key = (
            context_key_for_dm(author_id)
            if is_dm
            else context_key_for_guild(message.guild.id)
        )
        channel_name = "dm" if is_dm else message.channel.name
        guild_id = context_key if is_dm else str(message.guild.id)
        server_name = "DM" if is_dm else message.guild.name

        memory.append(
            key=context_key,
            channel=channel_name,
            author=message.author.display_name,
            author_id=author_id,
            content=message.content,
        )

        bot_was_mentioned = bool(client.user and client.user in message.mentions)

        if not await mirey.should_respond(
            content=message.content,
            context_key=context_key,
            bot_was_mentioned=bot_was_mentioned,
            is_dm=is_dm,
        ):
            print(f"Decided not to respond to message: {message.content} from {message.author.display_name} in channel {channel_name} (guild: {server_name})")
            return
        else:
            print(f"Responding to message: {message.content} from {message.author.display_name} in channel {channel_name} (guild: {server_name})")

        async with message.channel.typing():
            snapshot = memory.get_snapshot(context_key)
            response = await mirey.generate(
                AgentRequest(
                    content=message.content,
                    author_name=message.author.display_name,
                    author_id=author_id,
                    context_key=context_key,
                    channel_name=channel_name,
                    guild_id=guild_id,
                    server_name=server_name,
                    is_dm=is_dm,
                    context_snapshot=snapshot,
                )
            )
            if mirey.is_repeating_response(context_key, response.text):
                print(f"Detected repeated response for {context_key}; retrying once with anti-repeat instruction")
                response = await mirey.generate(
                    AgentRequest(
                        content=message.content,
                        author_name=message.author.display_name,
                        author_id=author_id,
                        context_key=context_key,
                        channel_name=channel_name,
                        guild_id=guild_id,
                        server_name=server_name,
                        is_dm=is_dm,
                        context_snapshot=snapshot,
                        repetition_instruction=(
                            "Your last several responses were identical. "
                            "Either respond meaningfully to the most recent message or stay silent. "
                            "Reply with [SKIP] if staying silent is correct."
                        ),
                    )
                )
            print(f"Generated response: {response.text} to {message.author.display_name} in channel {channel_name} (guild: {server_name})")

        if response.text.strip().lower() == "[skip]":
            print(f"Skipping response to {message.author.display_name} in channel {channel_name} (guild: {server_name})")
            return

        if response.text:
            await message.channel.send(content=response.text)
            memory.append_assistant(
                key=context_key,
                channel=channel_name,
                content=response.text,
            )

    async def guild_worker(context_key: str) -> None:
        queue = server_queues[context_key]
        metrics.set_worker_active(context_key, True)
        while True:
            message = await queue.get()
            try:
                msg_snapshot = MessageSnapshot(
                    author=message.author.display_name,
                    author_id=str(message.author.id),
                    author_avatar_url=message.author.display_avatar.url if message.author.display_avatar else None,
                    content=message.content[:100],
                    channel=message.channel.name,
                    message_id=str(message.id),
                    created_at=message.created_at.timestamp(),
                    enqueued_at=time.time(),
                )
                metrics.start_processing(context_key, msg_snapshot)
                
                await handle_message(message, is_dm=False)
                
                metrics.finish_processing(context_key)
            except Exception:
                log.exception("Failed to process queued guild message for context %s", context_key)
                metrics.finish_processing(context_key)
            finally:
                queue.task_done()
                metrics.update_queue_size(context_key, queue.qsize())

            await asyncio.sleep(0.5)

            async with worker_lock:
                if queue.empty():
                    metrics.set_worker_active(context_key, False)
                    active_workers.discard(context_key)
                    return

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
        author_id = str(message.author.id)

        if is_dm and int(author_id) != getattr(settings, 'owner_discord_id', 0):
            if getattr(settings, 'reply_to_rejected_dm', False):
                await message.channel.send("Not for you.")
            return

        if is_dm:
            await handle_message(message, is_dm=True)
            return

        context_key = context_key_for_guild(message.guild.id)
        
        msg_snapshot = MessageSnapshot(
            author=message.author.display_name,
            author_id=str(message.author.id),
            author_avatar_url=message.author.display_avatar.url if message.author.display_avatar else None,
            content=message.content[:100],
            channel=message.channel.name,
            message_id=str(message.id),
            created_at=message.created_at.timestamp(),
            enqueued_at=time.time(),
        )
        metrics.enqueue_message(context_key, msg_snapshot)
        
        async with worker_lock:
            await server_queues[context_key].put(message)
            metrics.update_queue_size(context_key, server_queues[context_key].qsize())
            if context_key not in active_workers:
                active_workers.add(context_key)
                asyncio.create_task(guild_worker(context_key))

    return client, metrics


async def main() -> None:
    bot_token = (
        getattr(settings, "discord_bot_token", None) 
        or getattr(settings, "discord_token", None)
        or os.environ.get("DISCORD_BOT_TOKEN") 
        or os.environ.get("DISCORD_TOKEN")
    )
    if not bot_token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required.")
        
    client, metrics = build_client()
    
    asyncio.create_task(run_observability_server(metrics))
    
    await client.start(bot_token)


if __name__ == "__main__":
    asyncio.run(main())