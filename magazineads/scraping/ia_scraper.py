"""Download magazine issues from the Internet Archive using the search API.

Instead of predicting exact URLs from volume/number patterns (fragile), this
scraper queries the IA search API to discover all items matching a magazine's
identifier prefix, then downloads the page images for each issue.

Requires the ``internetarchive`` Python package (``pip install internetarchive``).

Usage:
    # Download all Variety issues as JP2 zips
    python -m magazineads.scraping.ia_scraper \\
        --query "identifier:sim_variety_*" \\
        --output_dir /data/magazines/variety \\
        --format jp2

    # Download Time issues from 1945-1960
    python -m magazineads.scraping.ia_scraper \\
        --query "identifier:sim_time_* AND date:[1945-01-01 TO 1960-12-31]" \\
        --output_dir /data/magazines/time \\
        --format jp2

    # Use a preset magazine config
    python -m magazineads.scraping.ia_scraper \\
        --magazine variety \\
        --output_dir /data/magazines/variety

    # Dry run — list items without downloading
    python -m magazineads.scraping.ia_scraper \\
        --magazine esquire --dry-run

How it works:
    1. ``internetarchive.search_items(query)`` discovers all matching IA items
    2. For each item, we look for JP2 zip files (or PDF fallback)
    3. ``item.download()`` fetches the files with retry logic
    4. Downloaded zips are unpacked into per-issue directories of page images

This replaces the legacy template-based scrapers in ``magazineads.scraping.legacy/``,
which required per-magazine volume numbering logic and were prone to missing
issues whose identifiers didn't match the expected pattern.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import internetarchive as ia

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Magazine presets ──────────────────────────────────────────────────────────
# Each preset maps to an IA search query that discovers all issues for that
# magazine. The identifier prefix (e.g., "sim_variety_") is the stable part;
# the search API handles the rest.

MAGAZINE_PRESETS: dict[str, dict] = {
    # NOTE: Time magazine is excluded — its IA collection is not public.
    "esquire": {
        "query": "identifier:sim_esquire_*",
        "description": "Esquire magazine (monthly, 1933–2008)",
        "preferred_format": "jp2",
    },
    "life": {
        "query": "identifier:sim_life_*",
        "description": "Life magazine (weekly, 1936–1972)",
        "preferred_format": "jp2",
    },
    "forbes": {
        "query": "identifier:sim_forbes_*",
        "description": "Forbes magazine (biweekly, 1917–2000)",
        "preferred_format": "jp2",
    },
    "variety": {
        "query": "identifier:sim_variety_*",
        "description": "Variety magazine (weekly, 1905–1988)",
        "preferred_format": "jp2",
    },
}


def search_items(query: str, sorts: list[str] | None = None) -> list[dict]:
    """Search the Internet Archive and return item metadata.

    Args:
        query: IA search query (e.g., "identifier:sim_variety_*").
        sorts: Sort fields (default: identifier ascending).

    Returns:
        List of dicts with at least 'identifier' and 'title' keys.
    """
    if sorts is None:
        sorts = ["identifier asc"]

    logger.info("Searching IA: %s", query)
    results = []
    for item in ia.search_items(query, sorts=sorts):
        results.append(item)

    logger.info("Found %d items", len(results))
    return results


def _pick_download_files(item: ia.Item, preferred_format: str = "jp2") -> list[str]:
    """Select which files to download from an IA item.

    Prefers JP2 zip archives; falls back to the item PDF if no JP2 zip exists.

    Args:
        item: An ``internetarchive.Item``.
        preferred_format: "jp2" for JP2 zip archives, "pdf" for PDF.

    Returns:
        List of filenames to download (relative to the item).
    """
    files = item.files

    if preferred_format == "jp2":
        jp2_zips = [f["name"] for f in files if f["name"].endswith("_jp2.zip")]
        if jp2_zips:
            return jp2_zips

    # Fallback: main PDF (not the _text.pdf derivative)
    pdfs = [
        f["name"] for f in files
        if f["name"].endswith(".pdf") and not f["name"].endswith("_text.pdf")
    ]
    if pdfs:
        # Prefer the one matching the identifier
        identifier = item.identifier
        for pdf in pdfs:
            if pdf.startswith(identifier):
                return [pdf]
        return [pdfs[0]]

    return []


def download_item(
    identifier: str,
    output_dir: Path,
    preferred_format: str = "jp2",
    dry_run: bool = False,
    retries: int = 3,
    pause: float = 1.0,
) -> bool:
    """Download files for a single IA item.

    Args:
        identifier: IA item identifier (e.g., "sim_variety_1945-03-14_158_1").
        output_dir: Base output directory. Files go into ``output_dir/identifier/``.
        preferred_format: "jp2" or "pdf".
        dry_run: If True, log what would be downloaded without fetching.
        retries: Number of download retries per file.
        pause: Seconds between retries.

    Returns:
        True if files were downloaded (or would be in dry_run), False if skipped.
    """
    item_dir = output_dir / identifier
    item = ia.get_item(identifier)
    target_files = _pick_download_files(item, preferred_format)

    if not target_files:
        logger.warning("No downloadable files for %s", identifier)
        return False

    # Check if already downloaded
    if not dry_run and item_dir.exists():
        existing = {f.name for f in item_dir.iterdir()}
        if all(f in existing for f in target_files):
            logger.debug("Already downloaded: %s", identifier)
            return False

    if dry_run:
        for f in target_files:
            logger.info("[DRY RUN] Would download %s/%s", identifier, f)
        return True

    item_dir.mkdir(parents=True, exist_ok=True)

    for filename in target_files:
        dest = item_dir / filename
        if dest.exists():
            logger.debug("Skipping existing file: %s", dest)
            continue

        for attempt in range(1, retries + 1):
            try:
                logger.info("Downloading %s/%s (attempt %d)", identifier, filename, attempt)
                item.download(
                    files=[filename],
                    destdir=str(output_dir),
                    no_directory=False,
                    retries=retries,
                )
                break
            except Exception as e:
                logger.warning("Download failed: %s (attempt %d/%d)", e, attempt, retries)
                if attempt < retries:
                    time.sleep(pause * attempt)
                else:
                    logger.error("Giving up on %s/%s after %d attempts",
                                 identifier, filename, retries)
                    return False

    return True


def download_magazine(
    query: str,
    output_dir: Path,
    preferred_format: str = "jp2",
    dry_run: bool = False,
    max_items: int | None = None,
    retries: int = 3,
) -> dict:
    """Download all issues of a magazine matching the search query.

    Args:
        query: IA search query.
        output_dir: Base output directory.
        preferred_format: "jp2" or "pdf".
        dry_run: If True, only list items.
        max_items: Stop after this many items (for testing).
        retries: Download retries per file.

    Returns:
        Summary dict with counts of downloaded, skipped, and failed items.
    """
    items = search_items(query)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(items), "downloaded": 0, "skipped": 0, "failed": 0}

    for i, item_meta in enumerate(items):
        if max_items is not None and i >= max_items:
            logger.info("Reached max_items=%d, stopping", max_items)
            break

        identifier = item_meta["identifier"]
        try:
            downloaded = download_item(
                identifier, output_dir, preferred_format,
                dry_run=dry_run, retries=retries,
            )
            if downloaded:
                stats["downloaded"] += 1
            else:
                stats["skipped"] += 1
        except Exception:
            logger.exception("Error processing %s", identifier)
            stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download magazine issues from the Internet Archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Download all Variety issues
  %(prog)s --magazine variety --output_dir /data/magazines/variety

  # Custom search query with date range
  %(prog)s --query "identifier:sim_time_* AND date:[1945-01-01 TO 1960-12-31]" \\
           --output_dir /data/magazines/time_postwar

  # Dry run to see what would be downloaded
  %(prog)s --magazine life --dry-run

Available magazine presets: """ + ", ".join(sorted(MAGAZINE_PRESETS)),
    )
    parser.add_argument(
        "--magazine", choices=sorted(MAGAZINE_PRESETS),
        help="Magazine preset name (provides default query and format)",
    )
    parser.add_argument(
        "--query",
        help="Custom IA search query (overrides --magazine preset)",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Base directory for downloaded files",
    )
    parser.add_argument(
        "--format", dest="preferred_format", default="jp2",
        choices=["jp2", "pdf"],
        help="Preferred download format (default: jp2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List items without downloading",
    )
    parser.add_argument(
        "--max-items", type=int, default=None,
        help="Maximum number of items to process (for testing)",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Download retries per file (default: 3)",
    )
    args = parser.parse_args()

    if args.query:
        query = args.query
        fmt = args.preferred_format
    elif args.magazine:
        preset = MAGAZINE_PRESETS[args.magazine]
        query = preset["query"]
        fmt = args.preferred_format or preset["preferred_format"]
        logger.info("Using preset '%s': %s", args.magazine, preset["description"])
    else:
        parser.error("Either --magazine or --query is required")
        return

    stats = download_magazine(
        query=query,
        output_dir=args.output_dir,
        preferred_format=fmt,
        dry_run=args.dry_run,
        max_items=args.max_items,
        retries=args.retries,
    )

    print(f"\nDownload summary:")
    print(f"  Total items found: {stats['total']}")
    print(f"  Downloaded: {stats['downloaded']}")
    print(f"  Skipped (already exists): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")


if __name__ == "__main__":
    main()
