#!/usr/bin/env python3
"""
Scraper for U.S. News (and U.S. News & World Report) issues from the Internet Archive.

URL pattern (jp2 zip files):
    https://archive.org/download/{identifier}/{identifier}_jp2.zip

Because U.S. News is a WEEKLY magazine, identifiers use date strings rather than
simple YYYY-MM patterns (e.g., "1934-12-17"). This makes it impractical to generate
URLs from volume/number logic as monthly scrapers do. Instead, this scraper queries
the IA full-text search API to discover all matching identifiers, filters to the
requested year range, and downloads each.

IA identifier prefix for all issues:
    sim_u-s-news-weekly-special-issues_*

Date format in identifiers:
    Most issues: YYYY-MM-DD (e.g., sim_u-s-news-weekly-special-issues_1934-12-17_2_50)
    Index volumes: text format ending in year (e.g., ..._january-02-june-26-1942_12_index)
    Index volumes are excluded by default (--include-index to keep them).

Image format note:
    Each jp2 in the zip is a SINGLE PAGE (not a two-page spread).

Usage:
    python scrape_usnews.py --basedir /data/laviniad/magazines/usnews
    python scrape_usnews.py --basedir /tmp/usnews_test --dry-run
    python scrape_usnews.py --basedir /data/laviniad/magazines/usnews \\
        --first-year 1940 --last-year 1945
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from optparse import OptionParser

from magazineads.scraping.download_utils import download_file


MAGAZINE_SLUG = "sim_u-s-news-weekly-special-issues"
IA_SEARCH_URL = "https://archive.org/advancedsearch.php"


def extract_year(identifier: str) -> int | None:
    """
    Extract the publication year from a U.S. News identifier.

    Handles two formats:
      - YYYY-MM-DD prefix: sim_u-s-news-weekly-special-issues_1934-12-17_2_50
      - Text-date (index volumes): sim_u-s-news-weekly-special-issues_january-02-...-1942_12_index
    """
    parts = identifier.split('_')
    if len(parts) < 3:
        return None
    date_part = parts[2]
    if date_part[:4].isdigit():
        return int(date_part[:4])
    # Text format: year is at the end (e.g., "january-02-june-26-1942")
    if date_part[-4:].isdigit():
        return int(date_part[-4:])
    return None


def is_index_volume(identifier: str) -> bool:
    """Return True if this identifier is an index volume (not a regular issue)."""
    return identifier.endswith('_index')


def fetch_identifiers(first_year: int, last_year: int, include_index: bool) -> list[str]:
    """
    Query the Internet Archive search API for all U.S. News identifiers,
    then filter to those whose embedded year falls within [first_year, last_year].

    Returns a sorted list of identifier strings.
    """
    params = {
        'q': f'identifier:{MAGAZINE_SLUG}*',
        'fl[]': 'identifier',
        'rows': '10000',
        'output': 'json',
        'sort[]': 'identifier asc',
    }
    url = IA_SEARCH_URL + '?' + urllib.parse.urlencode(params, doseq=True)
    print(f"Querying IA search API...")
    print(f"  {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"ERROR: IA search API request failed: {e}")
        sys.exit(1)

    docs = data.get('response', {}).get('docs', [])
    identifiers = [doc['identifier'] for doc in docs if 'identifier' in doc]

    # Filter to the requested year range
    filtered = []
    for ident in identifiers:
        if not include_index and is_index_volume(ident):
            continue
        year = extract_year(ident)
        if year is not None and first_year <= year <= last_year:
            filtered.append(ident)

    print(f"Found {len(filtered)} matching identifiers (years {first_year}–{last_year}).")
    return sorted(filtered)


def build_url(identifier: str) -> str:
    """Build the jp2.zip download URL for a given IA identifier."""
    return f"https://archive.org/download/{identifier}/{identifier}_jp2.zip"


def build_filename(identifier: str) -> str:
    """Build the local output filename from the identifier."""
    return f"{identifier}_jp2.zip"


def main():
    usage = (
        "%prog [options]\n\n"
        "Download U.S. News / U.S. News & World Report issues from the Internet Archive.\n"
        "Queries the IA search API to discover issue identifiers, then downloads each zip.\n"
        "Single-page jp2 scans. Coverage targeted: 1933-1955 (~50 issues/year)."
    )
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str,
                      default='/data/laviniad/magazines/usnews',
                      help='Output directory: default=%default')
    parser.add_option('--first-year', type=int, default=1933,
                      help='First year to download (inclusive): default=%default')
    parser.add_option('--last-year', type=int, default=1955,
                      help='Last year to download (inclusive): default=%default')
    parser.add_option('--include-index', action='store_true', default=False,
                      help='Also download index volumes (excluded by default): default=%default')
    parser.add_option('--pause', type=int, default=5,
                      help='Seconds to pause between downloads: default=%default')
    parser.add_option('--max-failures', type=int, default=10,
                      help='Stop after this many consecutive failures: default=%default')
    parser.add_option('--dry-run', action='store_true', default=False,
                      help='Print URLs without downloading: default=%default')

    (options, args) = parser.parse_args()

    data_dir = options.basedir

    print(f"Scraping U.S. News / U.S. News & World Report")
    print(f"Years: {options.first_year} – {options.last_year}")
    print(f"Output: {data_dir}")
    if options.dry_run:
        print("DRY RUN — no files will be downloaded")
    print()

    identifiers = fetch_identifiers(
        options.first_year, options.last_year, options.include_index
    )

    if not identifiers:
        print("No identifiers found. Check the year range or IA availability.")
        sys.exit(0)

    if not options.dry_run:
        os.makedirs(data_dir, exist_ok=True)

    download_count = 0
    failure_count = 0
    consecutive_failures = 0

    for identifier in identifiers:
        url = build_url(identifier)
        filename = build_filename(identifier)
        outfile = os.path.join(data_dir, filename)

        if not options.dry_run and os.path.exists(outfile):
            print(f"Already downloaded: {filename}")
            continue

        if options.dry_run:
            print(f"[DRY RUN] {url}")
            print(f"          → {outfile}")
            continue

        try:
            success = download_file(
                url, outfile,
                max_tries=3,
                pause_length=0.5,
                overwrite=True
            )
            if success:
                download_count += 1
                consecutive_failures = 0
            else:
                print(f"Skipped (no overwrite): {filename}")
        except Exception as e:
            print(f"FAILED: {url}  ({e})")
            failure_count += 1
            consecutive_failures += 1
            if consecutive_failures > options.max_failures:
                print(f"Too many consecutive failures ({consecutive_failures}), stopping.")
                sys.exit(1)

        time.sleep(options.pause)

    print()
    print(f"Done. Downloads: {download_count}  Failures: {failure_count}")


if __name__ == '__main__':
    main()
