import json
import re
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from app.config import get_settings


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        pass

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            pass

    brace_match = re.search(r"\{.*\}", text, flags=re.S)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:  # noqa: BLE001
            return None

    return None


class LLMGateway:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _provider(self) -> str:
        provider = (self.settings.llm_provider or "anthropic").strip().lower()
        return provider

    def is_ready(self) -> bool:
        provider = self._provider()
        if provider == "openai":
            return bool(self.settings.openai_api_key)
        if provider == "anthropic":
            return bool(self.settings.anthropic_api_key)
        return False

    def generate_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1600,
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        if not self.is_ready():
            return None

        provider = self._provider()
        if provider == "openai":
            try:
                client = OpenAI(api_key=self.settings.openai_api_key)
                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
            except Exception:  # noqa: BLE001
                return None

            content = response.choices[0].message.content if response.choices else None
            raw_text = content.strip() if isinstance(content, str) else ""
            if not raw_text:
                return None

            return _extract_json_object(raw_text)

        if provider != "anthropic":
            return None

        try:
            client = Anthropic(api_key=self.settings.anthropic_api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )
        except Exception:  # noqa: BLE001
            return None

        text_blocks: list[str] = []
        for block in response.content:
            maybe_text = getattr(block, "text", None)
            if isinstance(maybe_text, str):
                text_blocks.append(maybe_text)

        raw_text = "\n".join(text_blocks).strip()
        if not raw_text:
            return None

        return _extract_json_object(raw_text)


# Backward-compatible alias used across existing modules.
ClaudeGateway = LLMGateway
