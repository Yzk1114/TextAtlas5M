"""CleanTextSynth: pure-text images on a clean background.

Provides both the original Chinese builder and a parallel zh/en builder.
"""

from .build import build_clean_text_synth_cn
from .build_parallel import build_clean_text_synth_parallel

__all__ = ["build_clean_text_synth_cn", "build_clean_text_synth_parallel"]
