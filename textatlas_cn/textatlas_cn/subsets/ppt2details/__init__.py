"""PPT2Details-CN: convert Chinese PPTX to images and generate fluent VLM captions."""

from .build import build_ppt2details_cn
from .build_parallel import build_ppt2details_parallel

__all__ = ["build_ppt2details_cn", "build_ppt2details_parallel"]
