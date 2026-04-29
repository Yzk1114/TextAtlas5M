"""TextVisionBlend-CN: synthetic interleaved image-text pages built via PyMuPDF."""

from .build import build_text_vision_blend_cn
from .build_parallel import build_text_vision_blend_parallel

__all__ = ["build_text_vision_blend_cn", "build_text_vision_blend_parallel"]
