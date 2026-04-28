"""Unified LLM / VLM client wrapper.

Supports OpenAI (gpt-4o), DashScope (Qwen2.5 family), DeepSeek, Zhipu (GLM-4).
Adds disk caching, retry/backoff, and image-input helpers for VLM calls.

Each provider needs the matching env var:
    OPENAI_API_KEY
    DASHSCOPE_API_KEY
    DEEPSEEK_API_KEY
    ZHIPUAI_API_KEY
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import diskcache
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    model: str
    provider: str


class LLMClient:
    """Thin abstraction over multiple Chinese-friendly LLM providers."""

    def __init__(
        self,
        provider: str = "dashscope",
        model: str = "qwen2.5-72b-instruct",
        cache_dir: str | Path | None = None,
        max_retries: int = 5,
        timeout: int = 60,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = (
            diskcache.Cache(str(cache_dir)) if cache_dir is not None else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        prompt: str,
        system: str | None = None,
        images: Sequence[str | Path] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        provider = provider or self.provider
        model = model or self.model
        cache_key = self._cache_key(provider, model, prompt, system, images, temperature, max_tokens, kwargs)
        if self.cache is not None and cache_key in self.cache:
            data = self.cache[cache_key]
            return LLMResponse(**data)

        text, raw = self._dispatch(provider, model, prompt, system, images, temperature, max_tokens, kwargs)
        resp = LLMResponse(text=text, raw=raw, model=model, provider=provider)

        if self.cache is not None:
            self.cache[cache_key] = resp.__dict__
        return resp

    # ------------------------------------------------------------------
    # Internal: provider dispatch
    # ------------------------------------------------------------------
    def _dispatch(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: str | None,
        images: Sequence[str | Path] | None,
        temperature: float,
        max_tokens: int,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if provider == "openai":
            return self._call_openai(model, prompt, system, images, temperature, max_tokens, kwargs)
        if provider == "dashscope":
            return self._call_dashscope(model, prompt, system, images, temperature, max_tokens, kwargs)
        if provider == "deepseek":
            return self._call_deepseek(model, prompt, system, images, temperature, max_tokens, kwargs)
        if provider == "zhipu":
            return self._call_zhipu(model, prompt, system, images, temperature, max_tokens, kwargs)
        raise ValueError(f"Unknown LLM provider: {provider}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    def _call_openai(self, model, prompt, system, images, temperature, max_tokens, kwargs):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=self.timeout)
        messages = self._build_messages(prompt, system, images)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or "", resp.model_dump()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    def _call_dashscope(self, model, prompt, system, images, temperature, max_tokens, kwargs):
        # DashScope (Aliyun) compatible path.
        import dashscope  # type: ignore

        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        messages = self._build_messages_dashscope(prompt, system, images)
        if images:
            from dashscope import MultiModalConversation
            resp = MultiModalConversation.call(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            content = resp.output.choices[0].message.content
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content)
            else:
                text = str(content)
        else:
            from dashscope import Generation
            resp = Generation.call(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, result_format="message"
            )
            text = resp.output.choices[0].message.content
        return text, dict(resp)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    def _call_deepseek(self, model, prompt, system, images, temperature, max_tokens, kwargs):
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
            timeout=self.timeout,
        )
        messages = self._build_messages(prompt, system, images=None)  # text-only
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or "", resp.model_dump()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    def _call_zhipu(self, model, prompt, system, images, temperature, max_tokens, kwargs):
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])
        messages = self._build_messages(prompt, system, images)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        return resp.choices[0].message.content or "", resp.model_dump()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_messages(self, prompt: str, system: str | None, images: Sequence[str | Path] | None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": _to_data_url(img)}})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _build_messages_dashscope(self, prompt: str, system: str | None, images: Sequence[str | Path] | None):
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"text": system}]})
        if images:
            content = [{"text": prompt}]
            for img in images:
                content.append({"image": str(img)})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": [{"text": prompt}]})
        return messages

    def _cache_key(self, *args: Any) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(args, sort_keys=True, default=str).encode("utf-8"))
        return h.hexdigest()


def _to_data_url(path: str | Path) -> str:
    p = Path(path)
    mime = "image/jpeg"
    if p.suffix.lower() in {".png"}:
        mime = "image/png"
    elif p.suffix.lower() in {".webp"}:
        mime = "image/webp"
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{data}"
