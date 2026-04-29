"""Translation + cross-lingual similarity utilities.

Default translator: GPT-4o / Qwen2.5-72B via :class:`LLMClient` with a
strict system prompt that forbids paraphrasing key phrases.
Optional offline backend: NLLB-200 distilled.

Cross-lingual similarity uses bge-m3 embeddings (preferred) or LaBSE.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .llm import LLMClient


Direction = Literal["zh2en", "en2zh"]


SYSTEM_TRANSLATE = (
    "You are an expert bilingual translator. Translate the user's message faithfully, "
    "preserving every entity, number, technical term, punctuation style, and line break. "
    "Do not summarize, do not add explanations, do not localize. Output ONLY the translation."
)


PROMPT_TEMPLATES: dict[Direction, str] = {
    "en2zh": (
        "Translate the following English text into fluent, faithful Simplified Chinese. "
        "Keep numbers, named entities, and abbreviations intact.\n\n"
        "===\n{text}\n==="
    ),
    "zh2en": (
        "请将下面这段简体中文严谨地翻译为流畅、忠实的英文，保留所有数字、专有名词与缩写。\n\n"
        "===\n{text}\n==="
    ),
}


@dataclass
class TranslationResult:
    text: str
    direction: Direction
    model: str
    provider: str
    raw: dict


class Translator:
    def __init__(
        self,
        provider: str = "dashscope",
        model: str = "qwen2.5-72b-instruct",
        cache_dir: str | None = None,
    ) -> None:
        self.client = LLMClient(provider=provider, model=model, cache_dir=cache_dir)

    def translate(self, text: str, direction: Direction, temperature: float = 0.2, max_tokens: int = 4096) -> TranslationResult:
        prompt = PROMPT_TEMPLATES[direction].format(text=text)
        resp = self.client.chat(
            prompt=prompt, system=SYSTEM_TRANSLATE, temperature=temperature, max_tokens=max_tokens,
        )
        return TranslationResult(text=resp.text.strip(), direction=direction, model=resp.model, provider=resp.provider, raw=resp.raw)


@lru_cache(maxsize=1)
def _bge_m3():
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer("BAAI/bge-m3")


def cross_lingual_similarity(zh: str, en: str) -> float:
    """Return cosine similarity in [-1, 1] using bge-m3 (multi-lingual)."""
    if not zh.strip() or not en.strip():
        return 0.0
    model = _bge_m3()
    embs = model.encode([zh, en], normalize_embeddings=True)
    return float((embs[0] * embs[1]).sum())


def back_translation_similarity(zh: str, en: str, translator: Translator) -> float:
    """Round-trip QA: translate en→zh then compare with original zh."""
    rt = translator.translate(en, direction="en2zh").text
    return cross_lingual_similarity(zh, rt)
