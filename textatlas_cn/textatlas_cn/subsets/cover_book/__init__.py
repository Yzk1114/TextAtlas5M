"""CoverBook-CN: Chinese book covers harvested from public catalogs (Douban / OpenLibrary-zh)."""

from .build import build_cover_book_cn
from .build_parallel import build_cover_book_parallel

__all__ = ["build_cover_book_cn", "build_cover_book_parallel"]
