"""
FIURI neural module building blocks.
"""

from .PyURI_module import (
    FIURIModule,
    FiuriDenseConn,
    FiuriSparseGJConn,
)

__all__ = ["FIURIModule", "FiuriDenseConn", "FiuriSparseGJConn"]
