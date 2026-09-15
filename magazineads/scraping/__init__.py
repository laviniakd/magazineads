"""Scraping infrastructure for downloading magazine issues from the Internet Archive.

Two approaches are provided:

``ia_scraper`` (recommended)
    Uses the ``internetarchive`` Python library to search for and download
    magazine issues by querying the IA search API. This is robust — it discovers
    items dynamically rather than predicting URLs from volume/number patterns.

``legacy/`` (preserved for provenance)
    The original template-based scrapers that were used to build the corpus
    described in the dataset paper. These construct download URLs by predicting
    the exact IA identifier from date, volume, and issue number. This approach
    is fragile and requires per-magazine volume numbering logic.

Both approaches download JP2 zip archives (preferred) or PDFs from the Internet
Archive, which are then converted to per-page JPEGs for the processing pipeline.
"""
