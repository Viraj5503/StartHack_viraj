import json
import re
from typing import Any

from anthropic import Anthropic

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


class ClaudeGateway:
    def __init__(self) -> None:
        self.settings = get_settings()

    def is_ready(self) -> bool:
        return bool(self.settings.anthropic_api_key)

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
