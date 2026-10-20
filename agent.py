from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from config import Settings, settings
from memory import MemoryStore
from personas import PeopleStore
from storage import Database

log = logging.getLogger("mirey")


@dataclass
class AgentRequest:
    content: str
    author_name: str
    author_id: str
    context_key: str
    channel_name: str
    guild_id: str
    server_name: str
    is_dm: bool = False
    context_snapshot: list[dict[str, Any]] | None = None
    repetition_instruction: str = ""


@dataclass
class AgentResponse:
    text: str


class MireyAgent:
    def __init__(
        self,
        memory: MemoryStore,
        people: PeopleStore,
        db: Database,
        config: Settings = settings,
    ) -> None:
        self.memory = memory
        self.people = people
        self.db = db
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key)

    async def should_respond(
        self,
        content: str,
        context_key: str,
        bot_was_mentioned: bool,
        is_dm: bool,
    ) -> bool:
        if is_dm or bot_was_mentioned or "mirey" in content.lower():
            return True
        return await self._classify_relevance(context_key, content)

    async def generate(self, request: AgentRequest) -> AgentResponse:
        snapshot = (
            request.context_snapshot
            if request.context_snapshot is not None
            else self.memory.get_snapshot(request.context_key)
        )
        
        system_instruction = self._system_prompt(request)
        history_text = self._history_text(request.context_key, snapshot=snapshot)
        
        prompt = f"{history_text}\n\n[#{request.channel_name}] {request.author_name}: {request.content}"
        if request.repetition_instruction:
            prompt += f"\n\nSystem instruction note: {request.repetition_instruction}"

        def _call_gemini():
            model_name = getattr(self.config, "gemini_model", None) or "gemini-3.8-flash"
            return self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )

        try:
            # Aumentado a 60 segundos para evitar falsos timeouts
            response = await asyncio.wait_for(asyncio.to_thread(_call_gemini), timeout=60.0)
            text = response.text.strip() if response.text else ""
            return AgentResponse(text=text)
        except asyncio.TimeoutError:
            log.warning("Timeout de Gemini API al generar respuesta.")
            return AgentResponse(text="¡Espera un momento...! Me tardé en conectar con la nube.")
        except Exception as exc:
            log.exception("Error llamando a Gemini API: %s", exc)
            return AgentResponse(text="(✿◠‿◠) me dio un mareo en la nube, probá de nuevo.")

    async def _classify_relevance(self, context_key: str, content: str) -> bool:
        history = self._history_text(context_key, limit=6)
        
        prompt = f"""Should I respond to the newest Discord message?
Answer YES or NO only.

Recent context:
{history}

Newest message:
{content}"""

        def _call_classifier():
            model_name = getattr(self.config, "gemini_model", None) or "gemini-3.8-flash"
            return self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=10,
                    temperature=0.1,
                ),
            )

        try:
            response = await asyncio.wait_for(asyncio.to_thread(_call_classifier), timeout=20.0)
            answer = response.text.strip().lower() if response.text else "no"
            return "yes" in answer
        except Exception:
            return True

    def is_repeating_response(self, context_key: str, new_response: str) -> bool:
        normalized_new = self._normalize_for_repeat_check(new_response)
        if not normalized_new:
            return False
        recent = [
            self._normalize_for_repeat_check(resp)
            for resp in self.memory.get_recent_assistant(context_key, 3)
        ]
        if len(recent) < 3:
            return False
        return all(resp == normalized_new for resp in recent)

    @staticmethod
    def _normalize_for_repeat_check(text: str) -> str:
        return "".join(
            char for char in text.lower().strip() if char.isalnum() or char.isspace()
        ).strip()

    def _system_prompt(self, request: AgentRequest) -> str:
        notes = self.people.raw
        try:
            with open('system_prompt.txt', encoding='utf-8') as f:
                base_prompt = f.read()
        except Exception:
            base_prompt = "You are mirey, a chat companion on Discord."

        return f"""You are mirey - a personal AI assistant running over Discord.

{base_prompt}
{request.repetition_instruction}

Personal notes context:
{notes}

Current server: {request.server_name}
Current channel: {request.channel_name}
"""

    def _history_text(
        self,
        context_key: str,
        limit: int = 15,
        snapshot: list[dict[str, Any]] | None = None,
    ) -> str:
        entries = (snapshot if snapshot is not None else self.memory.window(context_key))[-limit:]
        lines = []
        for entry in entries:
            role = entry.get("role")
            author = entry.get("author", "User")
            content = entry.get("content", "")
            if role == "assistant":
                lines.append(f"mirey: {content}")
            else:
                lines.append(f"{author}: {content}")
        return "\n".join(lines)