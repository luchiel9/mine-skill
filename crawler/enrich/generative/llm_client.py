from __future__ import annotations

import json
from typing import Any

import httpx

from crawler.enrich.models import LLMResponse


class LLMConfigurationError(ValueError):
    """Raised when a generative enrichment request lacks required AI config."""


class LLMRequestError(RuntimeError):
    """Raised when the configured model endpoint cannot return a response."""


class LLMEmptyResponseError(RuntimeError):
    """Raised when the model endpoint returns no usable content."""


class LLMClient:
    """Async client for OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        default_model: str = "",
        provider: str = "",
        openclaw_model: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.provider = provider.strip().lower()
        self.openclaw_model = openclaw_model.strip()
        self.timeout = timeout

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any]) -> LLMClient:
        return cls(
            base_url=str(model_config.get("base_url", "")),
            api_key=str(model_config.get("api_key", "")),
            default_model=str(model_config.get("model", "")),
            provider=str(model_config.get("provider", "")),
            openclaw_model=str(model_config.get("openclaw_model", "")),
            timeout=float(model_config.get("timeout", 60.0)),
        )

    async def complete(
        self,
        prompt: str,
        *,
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.2,
        system_prompt: str = "",
    ) -> LLMResponse:
        """Send a completion request to an OpenAI-compatible API."""
        resolved_model = model or self.default_model
        if not self.base_url or not resolved_model:
            raise LLMConfigurationError("AI configuration is incomplete")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url, payload = self._build_request(
            prompt=prompt,
            resolved_model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            headers=headers,
        )

        try:
            # trust_env=False: LLM endpoint is a local router, never use HTTP_PROXY
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            raise LLMRequestError("LLM request failed") from exc

        content = self._extract_content(data)
        if not content:
            raise LLMEmptyResponseError("LLM returned empty response")
        usage = self._extract_usage(data)
        return LLMResponse(
            content=content,
            model=data.get("model", resolved_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _build_request(
        self,
        *,
        prompt: str,
        resolved_model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, Any]]:
        if self._uses_openclaw_responses_api(resolved_model):
            if self.openclaw_model:
                headers["x-openclaw-model"] = self.openclaw_model

            input_items: list[dict[str, str]] = []
            if system_prompt:
                input_items.append({"type": "message", "role": "system", "content": system_prompt})
            input_items.append({"type": "message", "role": "user", "content": prompt})
            return (
                f"{self.base_url}/responses",
                {
                    "model": resolved_model,
                    "input": input_items,
                },
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return (
            f"{self.base_url}/chat/completions",
            {
                "model": resolved_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )

    def _uses_openclaw_responses_api(self, resolved_model: str) -> bool:
        return self.provider == "openclaw" or resolved_model.startswith("openclaw")

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        output = data.get("output")
        if isinstance(output, list):
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            if parts:
                return "".join(parts).strip()

        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage", {})
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if "input_tokens" in usage or "output_tokens" in usage:
            return {
                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
            }

        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }


def parse_json_response(content: str) -> dict[str, Any] | list[Any]:
    """Try to parse JSON from LLM response, handling markdown code blocks.

    Handles multiple fenced code blocks, nested fences, and bare JSON.
    Falls back to extracting the first ``{...}`` or ``[...]`` substring.
    """
    import re

    text = content.strip()

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (may be nested)
    fence_pattern = re.compile(r"^```\w*\s*\n(.*?)```\s*$", re.DOTALL)
    stripped = text
    for _ in range(3):
        m = fence_pattern.match(stripped)
        if not m:
            break
        stripped = m.group(1).strip()
    if stripped != text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Extract first complete JSON object or array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escape = False
        found_end = -1
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    found_end = i
                    break

        if found_end > start:
            try:
                return json.loads(text[start:found_end + 1])
            except json.JSONDecodeError:
                pass

        # Truncated JSON: track bracket stack and close
        if depth > 0 and start_char == '{':
            bracket_stack: list[str] = []
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if esc:
                    esc = False
                    continue
                if ch == '\\' and in_str:
                    esc = True
                    continue
                if ch == '"' and not esc:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch in ('{', '['):
                    bracket_stack.append(ch)
                elif ch == '}' and bracket_stack and bracket_stack[-1] == '{':
                    bracket_stack.pop()
                elif ch == ']' and bracket_stack and bracket_stack[-1] == '[':
                    bracket_stack.pop()

            if bracket_stack:
                truncated = text[start:].rstrip()
                # Trim incomplete trailing tokens
                for trim_char in (',', '"', ':'):
                    truncated = truncated.rstrip(trim_char)
                # Close all unclosed brackets in reverse order
                closing = {'[': ']', '{': '}'}
                truncated += ''.join(closing[b] for b in reversed(bracket_stack))
                try:
                    return json.loads(truncated)
                except json.JSONDecodeError:
                    pass

    return {"raw": content}
