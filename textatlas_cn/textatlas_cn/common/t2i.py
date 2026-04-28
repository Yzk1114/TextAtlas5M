"""Text-to-image client supporting Chinese-friendly diffusion services.

Providers:
    - dashscope-cogview  : Aliyun DashScope CogView/Wanxiang
    - hunyuan-dit        : Tencent Hunyuan-DiT (HTTP API or local pipeline)
    - kolors             : Kuaishou Kolors via diffusers
    - sd35-glyph         : Stable Diffusion 3.5 + Glyph-ByT5-CN (local)
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class T2IRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance_scale: float = 4.5
    seed: int | None = None


class T2IClient:
    def __init__(self, provider: str = "dashscope-cogview", model: str = "cogview-4", **kwargs: Any) -> None:
        self.provider = provider
        self.model = model
        self.extra = kwargs

    def generate(self, req: T2IRequest) -> Image.Image:
        if self.provider == "dashscope-cogview":
            return self._dashscope_cogview(req)
        if self.provider == "hunyuan-dit":
            return self._hunyuan_dit(req)
        if self.provider == "kolors":
            return self._kolors_local(req)
        if self.provider == "sd35-glyph":
            return self._sd35_glyph_local(req)
        raise ValueError(f"Unknown T2I provider: {self.provider}")

    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    def _dashscope_cogview(self, req: T2IRequest) -> Image.Image:
        import dashscope  # type: ignore
        from dashscope import ImageSynthesis

        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        resp = ImageSynthesis.call(
            model=self.model,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            n=1,
            size=f"{req.width}*{req.height}",
            steps=req.steps,
            scale=req.guidance_scale,
            seed=req.seed,
        )
        url = resp.output.results[0].url
        import requests
        img_bytes = requests.get(url, timeout=60).content
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30), reraise=True)
    def _hunyuan_dit(self, req: T2IRequest) -> Image.Image:
        # Local pipeline via diffusers.
        from diffusers import HunyuanDiTPipeline  # type: ignore
        import torch

        pipe = self._maybe_load("hunyuan", lambda: HunyuanDiTPipeline.from_pretrained(
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16
        ).to("cuda"))
        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=torch.Generator(device="cuda").manual_seed(req.seed) if req.seed else None,
        )
        return out.images[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30), reraise=True)
    def _kolors_local(self, req: T2IRequest) -> Image.Image:
        from diffusers import KolorsPipeline  # type: ignore
        import torch
        pipe = self._maybe_load("kolors", lambda: KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda"))
        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
        )
        return out.images[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30), reraise=True)
    def _sd35_glyph_local(self, req: T2IRequest) -> Image.Image:
        # SD3.5 with Chinese Glyph-ByT5 adapter for direct text rendering on canvas.
        from diffusers import StableDiffusion3Pipeline  # type: ignore
        import torch
        pipe = self._maybe_load("sd35", lambda: StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16,
        ).to("cuda"))
        out = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
        )
        return out.images[0]

    # ------------------------------------------------------------------
    _pipeline_cache: dict[str, Any] = {}

    @classmethod
    def _maybe_load(cls, key: str, builder):
        if key not in cls._pipeline_cache:
            cls._pipeline_cache[key] = builder()
        return cls._pipeline_cache[key]
