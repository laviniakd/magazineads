#!/usr/bin/env python3
"""
Unified Magazine Scraper

Consolidates 6 duplicate scraping scripts (1,135 LOC) into a single
configurable scraper with magazine-specific JSON configs.

Supports:
- Variety, The Nation, National Geographic, The Economist, Esquire, Esquire JP2
- Custom URL building for complex patterns
- Pluggable date incrementers and failure handlers
- Magazine-specific configurations via JSON files

Usage:
    python unified_scraper.py --magazine variety --first-date 2020-01-01 --last-date 2020-12-31
    python unified_scraper.py --config magazine_configs/esquire.json --first-date 1950-01-01
"""

import os
import sys
import json
import datetime as dt
import time
from pathlib import Path
from optparse import OptionParser
from typing import Dict, Any, Optional

# Import common utilities
from magazineads.scraping.download_utils import download_file, check_if_already_downloaded

# Import URL builders
from magazineads.scraping.legacy.url_builders import (
    TemplateURLBuilder,
    TheEconomistURLBuilder,
    EsquireURLBuilder
)


class UnifiedMagazineScraper:
    """
    Unified scraper for magazine downloads with pluggable components.

    Eliminates code duplication across 6 magazine-specific scripts by
    using configuration-driven approach with custom URL builders.
    """

    # Map custom URL builder names to classes
    URL_BUILDERS = {
        'the_economist': TheEconomistURLBuilder,
        'esquire': EsquireURLBuilder,
        None: TemplateURLBuilder  # Default for simple magazines
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scraper with magazine configuration.

        Args:
            config: Magazine configuration dictionary loaded from JSON
        """
        self.config = config
        self.magazine_name = config['name']
        self.display_name = config['display_name']

        # Initialize URL builder based on config
        custom_builder = config.get('custom_url_builder')
        builder_class = self.URL_BUILDERS.get(custom_builder, TemplateURLBuilder)
        self.url_builder = builder_class(config)

        # Initialize counters
        self.failure_counter = 0
        self.download_counter = 0

    @classmethod
    def from_config_file(cls, config_path: str) -> 'UnifiedMagazineScraper':
        """
        Create scraper instance from JSON configuration file.

        Args:
            config_path: Path to magazine configuration JSON file

        Returns:
            UnifiedMagazineScraper instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_magazine_name(cls, magazine_name: str) -> 'UnifiedMagazineScraper':
        """
        Create scraper instance from magazine name.

        Args:
            magazine_name: Name of magazine (e.g., 'variety', 'esquire')

        Returns:
            UnifiedMagazineScraper instance
        """
        # Find config file
        script_dir = Path(__file__).parent
        config_path = script_dir / 'magazine_configs' / f'{magazine_name}.json'

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Available magazines: variety, the_nation, national_geographic, "
                f"the_economist, esquire, esquire_jp2"
            )

        return cls.from_config_file(str(config_path))

    def increment_date(
        self,
        date: dt.date,
        volume: int,
        number: int
    ) -> tuple[dt.date, int, int]:
        """
        Increment date, volume, and number according to magazine's publication frequency.

        Args:
            date: Current issue date
            volume: Current volume number
            number: Current issue number

        Returns:
            Tuple of (next_date, next_volume, next_number)
        """
        frequency = self.config.get('publication_frequency', 'weekly')
        days = self.config.get('days_between_issues', 7)

        # Increment issue number
        next_number = number + 1

        # Handle special case: Esquire biweekly period
        if self.config.get('custom_url_builder') == 'esquire':
            days = self.url_builder.get_days_between_issues(date)

        # Increment date
        next_date = date + dt.timedelta(days=days)

        # Normalize to first of month if specified
        if self.config.get('normalize_to_first_of_month'):
            next_date = next_date.replace(day=1)

        # Handle volume increments
        next_volume = volume
        volume_rule = self.config.get('volume_increment_rule')

        if volume_rule == 'fixed_issues':
            # Volume increments after N issues (e.g., Variety: 13 issues per volume)
            issues_per_volume = self.config.get('issues_per_volume', 52)
            if next_number > issues_per_volume:
                next_number = 1
                next_volume = volume + 1

        elif volume_rule == 'yearly':
            # Volume increments each year (e.g., National Geographic)
            if next_date.year > date.year:
                next_volume = volume + 1
                next_number = 1

        elif volume_rule == 'biannual':
            # Volume increments in specific months (e.g., The Economist: Jan & July)
            increment_months = self.config.get('volume_increment_months', [1, 7])
            if next_date.month in increment_months and date.month not in increment_months:
                next_volume = volume + 1

        # volume_rule == 'manual' or None: no automatic increment

        return next_date, next_volume, next_number

    def handle_failure(
        self,
        date: dt.date,
        volume: int,
        number: int,
        data_dir: str,
        with_text: bool
    ) -> Optional[tuple[bool, int, int]]:
        """
        Handle download failure according to magazine's failure strategy.

        Args:
            date: Issue date
            volume: Volume number
            number: Issue number
            data_dir: Data directory
            with_text: Whether downloading text version

        Returns:
            Tuple of (retry_success, new_volume, new_number) or None if should stop
        """
        strategy = self.config.get('failure_strategy', 'counter')
        self.failure_counter += 1

        if strategy == 'immediate':
            # Stop immediately on first failure (The Economist)
            print(f"Failure strategy 'immediate': Stopping on first failure")
            return None

        elif strategy == 'retry_volume':
            # Try incrementing volume and resetting number to 1 (The Nation)
            print(f"Retrying with incremented volume...")
            new_volume = volume + 1
            new_number = 1

            date_str = date.strftime(self.config['date_format'])
            url = self.url_builder.build_url(date, new_volume, new_number, with_text)
            outfile_name = self.url_builder.build_filename(date, new_volume, new_number, with_text)
            outfile = os.path.join(data_dir, outfile_name)

            success = download_file(
                url,
                outfile,
                max_tries=self.config.get('max_retries', 3),
                pause_length=self.config.get('retry_pause', 0.5),
                overwrite=True
            )

            if success:
                return (True, new_volume, new_number)
            else:
                return (False, volume, number)

        elif strategy == 'counter':
            # Count failures and stop after max (Variety, Esquire)
            max_failures = self.config.get('max_failures', 10)
            if self.failure_counter > max_failures:
                print(f"Too many failed scrapes ({self.failure_counter}), stopping script.")
                return None

        return (False, volume, number)

    def scrape(
        self,
        first_date: str,
        last_date: str,
        first_volume: int,
        first_number: int,
        last_volume: int,
        last_number: int,
        data_dir: str,
        pause: int = 5,
        with_text: bool = False,
        dry_run: bool = False
    ):
        """
        Main scraping loop.

        Args:
            first_date: Starting date (YYYY-MM-DD or YYYY-MM)
            last_date: Ending date (YYYY-MM-DD or YYYY-MM)
            first_volume: Starting volume number
            first_number: Starting issue number
            last_volume: Ending volume number
            last_number: Ending issue number
            data_dir: Directory to save downloads
            pause: Seconds to pause between downloads
            with_text: Whether to download text-layer PDFs
            dry_run: If True, print URLs without downloading
        """
        # Parse dates
        date_format = self.config.get('date_format', '%Y-%m-%d')
        try:
            date = dt.datetime.strptime(first_date, date_format).date()
            end_date = dt.datetime.strptime(last_date, date_format).date()
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            print(f"Expected format: {date_format}")
            return

        # Initialize
        volume = first_volume
        number = first_number
        done = False

        # Create data directory if needed
        if not dry_run:
            os.makedirs(data_dir, exist_ok=True)

        print(f"Scraping {self.display_name}")
        print(f"From: {first_date} (vol {first_volume}, no {first_number})")
        print(f"To: {last_date} (vol {last_volume}, no {last_number})")
        print(f"Output: {data_dir}")
        if dry_run:
            print("DRY RUN MODE - No downloads will occur")
        print()

        # Main loop
        while not done:
            date_str = date.strftime(date_format)

            # Build URL and filename
            url = self.url_builder.build_url(date, volume, number, with_text)
            outfile_name = self.url_builder.build_filename(date, volume, number, with_text)
            outfile = os.path.join(data_dir, outfile_name)

            # Check if already downloaded
            if check_if_already_downloaded(data_dir, date_str, volume, number, with_text):
                print(f"Already downloaded: {outfile_name}")
            else:
                if dry_run:
                    print(f"[DRY RUN] Would download: {url}")
                    print(f"          Save to: {outfile}")
                else:
                    print(f"Downloading: {url}")
                    try:
                        success = download_file(
                            url,
                            outfile,
                            max_tries=self.config.get('max_retries', 3),
                            pause_length=self.config.get('retry_pause', 0.5),
                            overwrite=True
                        )

                        if success:
                            self.download_counter += 1
                        else:
                            # Handle failure
                            print(f"Failed on {date_str}, volume {volume}, number {number}")
                            result = self.handle_failure(date, volume, number, data_dir, with_text)

                            if result is None:
                                # Stop scraping
                                done = True
                                continue
                            elif result[0]:
                                # Retry succeeded, update volume/number
                                volume = result[1]
                                number = result[2]

                        time.sleep(pause)

                    except Exception as e:
                        print(f"Exception during download: {e}")
                        done = True
                        continue

            # Increment date/volume/number
            date, volume, number = self.increment_date(date, volume, number)

            # Check termination conditions
            if date > end_date or volume > last_volume or (volume == last_volume and number > last_number):
                done = True

        print()
        print(f"Scraping complete!")
        print(f"Downloads: {self.download_counter}")
        print(f"Failures: {self.failure_counter}")


def main():
    """Main CLI entry point."""
    usage = """
    %prog --magazine MAGAZINE [options]
    %prog --config CONFIG_FILE [options]

    Unified magazine scraper with pluggable configurations.

    Examples:
      %prog --magazine variety --first-date 2020-01-01 --last-date 2020-12-31
      %prog --config magazine_configs/esquire.json --first-date 1950-01-01 --first-number 1
      %prog --magazine the_economist --dry-run
    """

    parser = OptionParser(usage=usage)

    # Magazine selection
    parser.add_option('--magazine', type=str,
                      help='Magazine name (variety, the_nation, national_geographic, the_economist, esquire, esquire_jp2)')
    parser.add_option('--config', type=str,
                      help='Path to magazine configuration JSON file')

    # Date range
    parser.add_option('--first-date', type=str,
                      help='First date (format depends on magazine)')
    parser.add_option('--last-date', type=str,
                      help='Last date (format depends on magazine)')

    # Volume/number range
    parser.add_option('--first-volume', type=int,
                      help='First volume number')
    parser.add_option('--first-number', type=int,
                      help='First issue number')
    parser.add_option('--last-volume', type=int,
                      help='Last volume number')
    parser.add_option('--last-number', type=int,
                      help='Last issue number')

    # Output
    parser.add_option('--basedir', type=str,
                      help='Base directory for downloads')

    # Options
    parser.add_option('--pause', type=int, default=5,
                      help='Pause between downloads: default=%default')
    parser.add_option('--with-text', action="store_true", default=False,
                      help='Download PDFs with text layer: default=%default')
    parser.add_option('--dry-run', action="store_true", default=False,
                      help='Print URLs without downloading: default=%default')

    (options, args) = parser.parse_args()

    # Load configuration
    if options.config:
        scraper = UnifiedMagazineScraper.from_config_file(options.config)
    elif options.magazine:
        scraper = UnifiedMagazineScraper.from_magazine_name(options.magazine)
    else:
        parser.error("Must specify either --magazine or --config")

    # Use config defaults or CLI overrides
    config = scraper.config

    first_date = options.first_date or config.get('default_first_date')
    last_date = options.last_date or config.get('default_last_date')
    first_volume = options.first_volume or config.get('default_first_volume', 1)
    first_number = options.first_number or config.get('default_first_number', 1)
    last_volume = options.last_volume or config.get('default_last_volume', 999)
    last_number = options.last_number or config.get('default_last_number', 52)
    data_dir = options.basedir or config.get('default_basedir')

    if not first_date or not last_date:
        parser.error("Must specify --first-date and --last-date")

    if not data_dir:
        parser.error("Must specify --basedir")

    # Run scraper
    scraper.scrape(
        first_date=first_date,
        last_date=last_date,
        first_volume=first_volume,
        first_number=first_number,
        last_volume=last_volume,
        last_number=last_number,
        data_dir=data_dir,
        pause=options.pause,
        with_text=options.with_text,
        dry_run=options.dry_run
    )


if __name__ == '__main__':
    main()
