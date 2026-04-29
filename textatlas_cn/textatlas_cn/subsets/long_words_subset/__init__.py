"""LongWordsSubset-CN: long-text Chinese samples filtered from public OCR datasets."""

from .build import build_long_words_subset_cn
from .build_parallel import build_long_words_subset_parallel

__all__ = ["build_long_words_subset_cn", "build_long_words_subset_parallel"]
