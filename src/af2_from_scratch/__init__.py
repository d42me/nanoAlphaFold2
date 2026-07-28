"""Educational, compact implementation of the AlphaFold 2 architecture."""

from .config import AF2Config
from .model import AlphaFold2FromScratch

__all__ = ["AF2Config", "AlphaFold2FromScratch"]
