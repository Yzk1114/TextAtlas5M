"""TextScenesHQ-CN: high-quality Chinese real-world dense-text scene images."""

from .build import build_text_scenes_hq_cn
from .build_parallel import build_text_scenes_hq_parallel

__all__ = ["build_text_scenes_hq_cn", "build_text_scenes_hq_parallel"]
