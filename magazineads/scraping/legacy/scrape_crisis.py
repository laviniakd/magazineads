#!/usr/bin/env python3
"""
Scraper for The Crisis issues from the Internet Archive.

URL pattern (jp2 zip files):
    https://archive.org/download/sim_crisis_the-crisis_{YYYY-MM}_{vol}_{no}/
        sim_crisis_the-crisis_{YYYY-MM}_{vol}_{no}_jp2.zip

Image format note:
    Each jp2 in the zip is a SINGLE PAGE (not a two-page spread).

Volume numbering:
    November and May each start a new volume (issue No. 1).
    Nov 1910 = Vol 1, No. 1 (first issue).
    Volume increments each November and each May.

Usage:
    python scrape_crisis.py --basedir /data/laviniad/magazines/crisis
    python scrape_crisis.py --basedir /tmp/crisis_test --dry-run
"""

import os
import sys
import time
import datetime as dt
from optparse import OptionParser

from magazineads.scraping.download_utils import download_file


ARCHIVE_BASE = "https://archive.org/download/sim_crisis_the-crisis_"
MAGAZINE_SLUG = "sim_crisis_the-crisis"
#MAGAZINE_SLUG_2 = "sim_crisis" # used occasionally, unclear


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

    November and May each start a new volume (issue No. 1), so when advancing
    into either month we increment the volume and reset the issue number to 1.
    """
    # Move to first of next month
    if date.month == 12:
        next_date = date.replace(year=date.year + 1, month=1, day=1)
    else:
        next_date = date.replace(month=date.month + 1, day=1)

    # November and May each start a new volume
    if next_date.year >= 1911 and next_date.month > 11:
        if next_date.month in (5, 11):
            return next_date, volume + 1, 1
        else:
            return next_date, volume, number + 1
    else:
        if next_date.month == 1:
            return next_date, volume + 1, 1
        else:
            return next_date, volume, number + 1


def main():
    usage = (
        "%prog [options]\n\n"
        "Download The Crisis issues from the Internet Archive (jp2.zip).\n"
        "First issue: November 1910 (vol 1, no 1).\n"
        "Volumes run May–October and November–April; May and November are always issue No. 1."
    )
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str,
                      default='/data/laviniad/magazines/crisis',
                      help='Output directory: default=%default')
    parser.add_option('--first-date', type=str, default='1910-11',
                      help='First issue date (YYYY-MM): default=%default')
    parser.add_option('--first-volume', type=int, default=1,
                      help='Volume number of first issue: default=%default')
    parser.add_option('--first-number', type=int, default=1,
                      help='Issue number within volume of first issue: default=%default')
    parser.add_option('--last-date', type=str, default='1960-12',
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

    print(f"Scraping The Crisis")
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
