"""Abstract base class for OCR extractors."""

from typing import Optional, Dict, Any
from PIL import Image


class AbstractOCRExtractor:
    """Base class for OCR extraction.

    All OCR extractors implement extract_text() for plain text retrieval.
    Extractors with rich data (e.g. surya) also implement extract_metadata()
    for bounding boxes, confidence scores, etc.
    """

    def extract_text(self, image_path: str, image: Optional[Image.Image] = None) -> str:
        """Return plain text for the given image.

        This is the primary interface used by classifier datasets.

        Args:
            image_path: Path to the image file.
            image: Optional pre-loaded PIL Image (for live OCR engines).

        Returns:
            Extracted text as a string, or "" if unavailable.
        """
        raise NotImplementedError

    def extract_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Return rich OCR metadata if available.

        Includes bounding boxes, confidence scores, character-level data, etc.
        Returns None by default; subclasses with rich data override this.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary with OCR metadata, or None if not supported.
        """
        return None
