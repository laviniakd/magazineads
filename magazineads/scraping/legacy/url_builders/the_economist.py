"""
Custom URL builder for The Economist magazine.

The Economist has date-conditional URL patterns:
- Before 1951-07-07: Uses {date}_{volume}_{number}
- From 1951-07-07 onward: Uses {date}_{volume} (no issue number)
"""

import datetime as dt
from typing import Dict, Any


class TheEconomistURLBuilder:
    """
    URL builder for The Economist with date-conditional patterns.

    The URL structure changed on 1951-07-07:
    - Old pattern: sim_economist_2020-01-15_100_5
    - New pattern: sim_economist_2020-01-15_100 (no issue number)
    """

    PATTERN_CHANGE_DATE = dt.date(1951, 7, 7)

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize URL builder with magazine configuration.

        Args:
            config: Magazine configuration dictionary
        """
        self.config = config
        self.base_url = config['base_url']
        self.date_format = config['date_format']
        self.magazine_name = config['name']

    def build_url(
        self,
        date: dt.date,
        volume: int,
        number: int,
        with_text: bool = False
    ) -> str:
        """
        Build download URL with date-conditional pattern.

        Args:
            date: Issue date
            volume: Volume number
            number: Issue number (ignored for dates >= 1951-07-07)
            with_text: Whether to get PDF with text layer

        Returns:
            Complete download URL

        Example:
            Before 1951-07-07:
            https://archive.org/download/sim_economist_1950-01-15_100_5/sim_economist_1950-01-15_100_5.pdf

            After 1951-07-07:
            https://archive.org/download/sim_economist_2020-01-15_437/sim_economist_2020-01-15_437.pdf
        """
        date_str = date.strftime(self.date_format)

        # Determine pattern based on date
        if date < self.PATTERN_CHANGE_DATE:
            # Old pattern: includes issue number
            issue_id = f"{date_str}_{volume}_{number}"
            file_path = f"sim_economist_{date_str}_{volume}_{number}"
        else:
            # New pattern: no issue number
            issue_id = f"{date_str}_{volume}"
            file_path = f"sim_economist_{date_str}_{volume}"

        # Build full URL
        url = f"{self.base_url}{issue_id}/{file_path}"

        # Add suffix
        if with_text:
            url += "_text.pdf"
        else:
            url += ".pdf"

        return url

    def build_filename(
        self,
        date: dt.date,
        volume: int,
        number: int,
        with_text: bool = False
    ) -> str:
        """
        Build local filename for downloaded issue.

        Args:
            date: Issue date
            volume: Volume number
            number: Issue number
            with_text: Whether this is a text-layer PDF

        Returns:
            Filename for local storage
        """
        date_str = date.strftime(self.date_format)
        filename = f"{date_str}-{volume}-{number}"

        if with_text:
            filename += "_text.pdf"
        else:
            filename += ".pdf"

        return filename
