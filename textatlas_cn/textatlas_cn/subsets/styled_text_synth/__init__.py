"""StyledTextSynth-CN: T2I scene + LLM text + bbox/quad rendering."""

from .build import build_styled_text_synth_cn
from .build_parallel import build_styled_text_synth_parallel

__all__ = ["build_styled_text_synth_cn", "build_styled_text_synth_parallel"]
