#!/usr/bin/env python3
"""
Scraper for Cosmopolitan issues from the Internet Archive.

URL pattern (jp2 zip files):
    https://archive.org/download/sim_cosmopolitan_{YYYY-MM}_{vol}_{no}/
        sim_cosmopolitan_{YYYY-MM}_{vol}_{no}_jp2.zip

Image format note:
    Each jp2 in the zip is a SINGLE PAGE (not a two-page spread).

Volume numbering:
    Volumes are semi-annual (6 issues each).
    New volumes start in November and May (No. 1 = November or May).
    Jan 1890 = Vol 8, No. 3 (first known IA issue).
    Jan 1891 = Vol 10, No. 3 confirms semi-annual cadence.

Usage:
    python scrape_cosmo.py --basedir /data/laviniad/magazines/cosmo
    python scrape_cosmo.py --basedir /tmp/cosmo_test --dry-run
"""

import os
import sys
import time
import datetime as dt
from optparse import OptionParser

from magazineads.scraping.download_utils import download_file


ARCHIVE_BASE = "https://archive.org/download/sim_cosmopolitan_"
MAGAZINE_SLUG = "sim_cosmopolitan"


def build_url(date: dt.date, volume: int, number: int) -> str:
    """Build the Internet Archive jp2.zip URL for a given issue."""
    date_str = date.strftime('%Y-%m')
    identifier = f"{MAGAZINE_SLUG}_{date_str}_{volume}_{number}"
    return f"{ARCHIVE_BASE}{date_str}_{volume}_{number}/{identifier}_jp2.zip"


def build_filename(date: dt.date, volume: int, number: int) -> str:
    """Build the local output filename."""
    date_str = date.strftime('%Y-%m')
    return f"{date_str}-{volume}-{number}_jp2.zip"


def next_issue(date: dt.date, volume: int, number: int):
    """
    Advance to the next monthly issue.

    Volumes are semi-annual (6 issues each): new volumes start in November
    and May. Jan 1890 = vol 8 no 3; Jan 1891 = vol 10 no 3 confirms this.
    """
    # Move to first of next month
    if date.month == 12:
        next_date = date.replace(year=date.year + 1, month=1, day=1)
    else:
        next_date = date.replace(month=date.month + 1, day=1)

    # November and May each start a new volume
    if next_date.month in (11, 5):
        return next_date, volume + 1, 1
    else:
        return next_date, volume, number + 1


def main():
    usage = (
        "%prog [options]\n\n"
        "Download Cosmopolitan issues from the Internet Archive (jp2.zip).\n"
        "First known IA issue: January 1890 (vol 8, no 3).\n"
        "Volumes are semi-annual (6 issues); new volumes start in November and May."
    )
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str,
                      default='/data/laviniad/magazines/cosmo',
                      help='Output directory: default=%default')
    parser.add_option('--first-date', type=str, default='1890-01',
                      help='First issue date (YYYY-MM): default=%default')
    parser.add_option('--first-volume', type=int, default=8,
                      help='Volume number of first issue: default=%default')
    parser.add_option('--first-number', type=int, default=3,
                      help='Issue number within volume of first issue: default=%default')
    parser.add_option('--last-date', type=str, default='1930-12',
                      help='Last issue date (YYYY-MM, inclusive): default=%default')
    parser.add_option('--pause', type=int, default=5,
                      help='Seconds to pause between downloads: default=%default')
    parser.add_option('--max-failures', type=int, default=10,
                      help='Stop after this many consecutive failures: default=%default')
    parser.add_option('--dry-run', action='store_true', default=False,
                      help='Print URLs without downloading: default=%default')

    (options, args) = parser.parse_args()

    # Parse dates
    try:
        date = dt.datetime.strptime(options.first_date, '%Y-%m').date().replace(day=1)
        end_date = dt.datetime.strptime(options.last_date, '%Y-%m').date().replace(day=1)
    except ValueError as e:
        parser.error(f"Date parse error: {e}. Expected YYYY-MM format.")

    volume = options.first_volume
    number = options.first_number
    data_dir = options.basedir

    print(f"Scraping Cosmopolitan (Cosmo)")
    print(f"From: {options.first_date} (vol {volume}, no {number})")
    print(f"To:   {options.last_date}")
    print(f"Output: {data_dir}")
    if options.dry_run:
        print("DRY RUN — no files will be downloaded")
    print()

    if not options.dry_run:
        os.makedirs(data_dir, exist_ok=True)

    download_count = 0
    failure_count = 0
    consecutive_failures = 0

    while date <= end_date:
        url = build_url(date, volume, number)
        filename = build_filename(date, volume, number)
        outfile = os.path.join(data_dir, filename)

        if not options.dry_run and os.path.exists(outfile):
            print(f"Already downloaded: {filename}")
        else:
            if options.dry_run:
                print(f"[DRY RUN] {url}")
                print(f"          → {outfile}")
            else:
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

        date, volume, number = next_issue(date, volume, number)

    print()
    print(f"Done. Downloads: {download_count}  Failures: {failure_count}")


if __name__ == '__main__':
    main()
