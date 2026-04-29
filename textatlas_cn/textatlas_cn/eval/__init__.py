"""TextAtlasEval-CN: 4×1000 stratified test set construction."""

from .build_eval import build_textatlas_eval_cn
from .build_eval_parallel import build_textatlas_eval_parallel

__all__ = ["build_textatlas_eval_cn", "build_textatlas_eval_parallel"]
