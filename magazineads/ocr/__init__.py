"""
OCR extraction system for magazine page analysis.

Provides the Surya OCR extractor for reading precomputed OCR results,
plus batch precomputation scripts.

Extractors:
    SuryaOCRExtractor - Reads precomputed surya output directories

Precompute scripts (in magazineads.ocr.precompute):
    surya_batch - GPU batch processing via surya Python API
"""

from magazineads.ocr.base import AbstractOCRExtractor
from magazineads.ocr.surya_extractor import SuryaOCRExtractor

__all__ = [
    "AbstractOCRExtractor",
    "SuryaOCRExtractor",
]
