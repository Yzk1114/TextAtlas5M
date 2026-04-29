"""Paper2Text-CN: Chinese academic PDFs parsed by PyMuPDF."""

from .build import build_paper2text_cn
from .build_parallel import build_paper2text_parallel

__all__ = ["build_paper2text_cn", "build_paper2text_parallel"]
