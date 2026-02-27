"""
LLM Client — thin OpenAI-compatible chat completion wrapper.

Default endpoint: http://127.0.0.1:1234/v1  (LM Studio / any local server)
Default model:    qwen3.5-397b-a17b

This model is a thinking/reasoning model that wraps its chain-of-thought in
<think>...</think> tags before the actual answer.  complete() strips those
tags automatically so callers always receive only the final response text.

Falls back gracefully (returns None) on any network or API error, so the
rest of the optimizer continues to work even when the LLM is offline.
"""
from __future__ import annotations

import json
import re

import requests
from loguru import logger

from config.settings import LLM_BASE_URL, LLM_MODEL

# Strip <think>...</think> (including Qwen3 "Thinking Process:" variant)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning blocks; return the final answer."""
    stripped = _THINK_RE.sub("", text).strip()
    return stripped if stripped else text  # fallback: return original if nothing left


class LLMClient:
    """Minimal stateless wrapper around an OpenAI-compatible /v1 endpoint."""

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL,
        # 300 s — optimizer runs overnight, slow is fine; 397B MoE ≈ 20-60 s/call
        timeout: int = 300,
    ) -> None:
        self._url     = base_url.rstrip("/") + "/chat/completions"
        self._health  = base_url.rstrip("/") + "/models"
        self._model   = model
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict],
        # 3000 tokens to give the thinking model enough room before its answer
        max_tokens: int = 3000,
        temperature: float = 0.2,
    ) -> str | None:
        """
        Send a chat completion request.
        Returns the assistant reply (with <think> blocks stripped), or None on failure.
        """
        payload = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(self._url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            answer = _strip_thinking(raw)
            logger.debug(f"[LLM] reply ({len(answer)} chars): {answer[:80]}…")
            return answer
        except Exception as exc:
            logger.warning(f"[LLM] complete() failed: {exc}")
            return None

    def complete_json(
        self,
        messages: list[dict],
        max_tokens: int = 3000,
        temperature: float = 0.1,
    ) -> dict | None:
        """
        Like complete(), but parses the first JSON object from the reply.
        Returns None if the LLM is offline or the reply contains no valid JSON.
        """
        reply = self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        if not reply:
            return None
        match = re.search(r"\{[\s\S]*\}", reply)
        if not match:
            logger.debug(f"[LLM] No JSON found in reply: {reply[:120]}")
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.debug(f"[LLM] JSON parse error: {exc} — raw: {reply[:120]}")
            return None

    def available(self) -> bool:
        """Quick reachability check — True if the server responds within 5 s."""
        try:
            requests.get(self._health, timeout=5)
            return True
        except Exception:
            return False
