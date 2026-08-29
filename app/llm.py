"""Anthropic client wrapper: retries, usage accumulation, structured output.

Two behaviours here are consequences of measurements against the proxy:

1. `messages.count_tokens` returns 404, so all token accounting comes from
   `response.usage`. There is no pre-flight token estimate anywhere in this
   codebase by design.

2. `output_config.format` (structured outputs) came back wrapped in a markdown
   fence rather than schema-constrained. Forced `tool_choice` returned clean,
   already-parsed JSON. So `structured()` below uses forced tool_choice.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from . import config


@dataclass
class Usage:
    """Accumulated token usage and cost across one or more calls."""

    in_tok: int = 0
    out_tok: int = 0
    cache_write: int = 0
    cache_read: int = 0
    embed_tok: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add_response(self, model: str, resp: Any) -> None:
        u = resp.usage
        in_tok = getattr(u, "input_tokens", 0) or 0
        out_tok = getattr(u, "output_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        self.in_tok += in_tok
        self.out_tok += out_tok
        self.cache_write += cw
        self.cache_read += cr
        self.calls += 1
        self.cost_usd += config.price_for(model).cost(in_tok, out_tok, cw, cr)

    def add_embedding(self, tokens: int) -> None:
        self.embed_tok += tokens
        self.cost_usd += config.embed_cost(tokens)

    def merge(self, other: "Usage") -> None:
        self.in_tok += other.in_tok
        self.out_tok += other.out_tok
        self.cache_write += other.cache_write
        self.cache_read += other.cache_read
        self.embed_tok += other.embed_tok
        self.cost_usd += other.cost_usd
        self.calls += other.calls

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens served from cache."""
        total = self.cache_read + self.cache_write + self.in_tok
        return self.cache_read / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "in_tok": self.in_tok,
            "out_tok": self.out_tok,
            "cache_write": self.cache_write,
            "cache_read": self.cache_read,
            "embed_tok": self.embed_tok,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "cache_hit_ratio": round(self.cache_hit_ratio, 4),
        }


class LLMError(RuntimeError):
    pass


# Retried: transient proxy failures. The proxy returned 503 model_not_found for
# an unavailable model id -- that is NOT transient, so 503 is retried only a
# couple of times and then surfaced with the model name in the message.
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 529}


class LLM:
    def __init__(self, max_retries: int = 3) -> None:
        config.require_credentials()
        # Pass credentials explicitly: the operator may have set
        # ANTHROPIC_AUTH_TOKEN instead of ANTHROPIC_API_KEY, and the SDK treats
        # them differently. config resolved whichever exists.
        self._client = anthropic.Anthropic(
            auth_token=config.AUTH_TOKEN,
            base_url=config.BASE_URL,
            max_retries=0,  # we own the retry loop so we can log/backoff
        )
        self.max_retries = max_retries

    # --- raw call ---------------------------------------------------------

    def call(self, *, model: str, usage: Usage | None = None, **kwargs: Any) -> Any:
        """One Messages API call with backoff. Accumulates into `usage`."""
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(model=model, **kwargs)
                if usage is not None:
                    usage.add_response(model, resp)
                return resp
            except anthropic.APIStatusError as e:
                last = e
                if e.status_code not in _RETRY_STATUS:
                    raise LLMError(f"{model}: {e.status_code} {e}") from e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2**attempt + random.random(), 8.0))
            except anthropic.APIConnectionError as e:
                last = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2**attempt + random.random(), 8.0))
        raise LLMError(
            f"{model}: exhausted {self.max_retries} attempts. Last error: {last}"
        ) from last

    # --- structured output ------------------------------------------------

    def structured(
        self,
        *,
        model: str,
        tool: dict,
        messages: list[dict],
        system: Any | None = None,
        max_tokens: int = 2000,
        usage: Usage | None = None,
    ) -> dict:
        """Get schema-shaped JSON back via forced tool_choice.

        Returns the tool's parsed `input` dict. Raises if the model somehow
        returned no tool_use block (never observed, but the caller should not
        have to guard).
        """
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool["name"]},
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        resp = self.call(model=model, usage=usage, **kwargs)
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool["name"]:
                return dict(block.input)
        raise LLMError(
            f"{model}: forced tool_choice for {tool['name']!r} returned no tool_use "
            f"block (stop_reason={resp.stop_reason})"
        )

    # --- preflight --------------------------------------------------------

    def preflight(self, model: str) -> tuple[bool, str]:
        """Cheap liveness probe. Used by the CLI/web to fail early and clearly."""
        try:
            self.call(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": "Reply with: OK"}],
            )
            return True, "ok"
        except Exception as e:  # noqa: BLE001 - surfacing any failure verbatim
            return False, str(e)[:200]


_shared: LLM | None = None


def shared() -> LLM:
    """Process-wide client. Reused so connection setup is not repeated."""
    global _shared
    if _shared is None:
        _shared = LLM()
    return _shared
