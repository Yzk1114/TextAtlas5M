"""PPT2Structured-CN: bbox-level structure of academic-style Chinese slides."""

from .build import build_ppt2structured_cn
from .build_parallel import build_ppt2structured_parallel

__all__ = ["build_ppt2structured_cn", "build_ppt2structured_parallel"]
