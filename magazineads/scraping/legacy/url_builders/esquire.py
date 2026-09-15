"""
Custom URL builder for Esquire magazine.

Esquire has complex URL patterns with multiple format changes:
1. Old format monthly (before 1978-04 and after 1979-07)
2. Old format biweekly (1978-04 to 1979-07)
3. New format (modern issues)
"""

import datetime as dt
from typing import Dict, Any


class EsquireURLBuilder:
    """
    URL builder for Esquire magazine with period-dependent formats.

    Three different URL patterns:
    1. Old format, monthly: Esquire-Magazine-2020-01/Esquire-Magazine-2020-01.pdf
    2. Old format, biweekly: Esquire-Magazine-2020-01-15/Esquire-Magazine-2020-01-15.pdf
    3. New format: Esquire_January_2020_USA/Esquire_January_2020_USA.pdf
    """

    BIWEEKLY_START = dt.date(1978, 4, 1)
    BIWEEKLY_END = dt.date(1979, 7, 31)

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize URL builder with magazine configuration.

        Args:
            config: Magazine configuration dictionary
        """
        self.config = config
        self.base_url = config['base_url']
        self.old_format = config.get('default_old_format', True)
        self.file_format = config.get('file_format', 'pdf')  # 'pdf' or 'jp2.zip'

    def _is_in_biweekly_period(self, date: dt.date) -> bool:
        """Check if date falls in the biweekly period (1978-04 to 1979-07)."""
        return self.BIWEEKLY_START <= date <= self.BIWEEKLY_END

    def _is_monthly_period(self, date: dt.date) -> bool:
        """Check if date is in monthly publication period."""
        return (date.year < 1978 or
                (date.year == 1978 and date.month < 4) or
                (date.year == 1979 and date.month > 7) or
                date.year > 1979)

    def build_url(
        self,
        date: dt.date,
        volume: int,
        number: int,
        with_text: bool = False,
        old_format: bool = None
    ) -> str:
        """
        Build download URL with format-dependent pattern.

        Args:
            date: Issue date
            volume: Volume number (not used - Esquire uses 'x' placeholder)
            number: Issue number
            with_text: Whether to get PDF with text layer
            old_format: Override for old/new format (None = use default)

        Returns:
            Complete download URL

        Examples:
            Old format monthly:
            https://archive.org/download/Esquire-Magazine-2020-01/Esquire-Magazine-2020-01.pdf

            Old format biweekly:
            https://archive.org/download/Esquire-Magazine-2020-01-15/Esquire-Magazine-2020-01-15.pdf

            New format:
            https://archive.org/download/Esquire_January_2020_USA/Esquire_January_2020_USA.pdf
        """
        if old_format is None:
            old_format = self.old_format

        if old_format and self._is_monthly_period(date):
            # Old format, monthly period
            date_str = date.strftime('%Y-%m')
            id_str = f"-Magazine-{date_str}"
            path_str = f"Esquire{id_str}"

        elif old_format and self._is_in_biweekly_period(date):
            # Old format, biweekly period
            date_str = date.strftime('%Y-%m-%d')
            id_str = f"-Magazine-{date_str}"
            path_str = f"Esquire{id_str}"

        else:
            # New format
            date_str = date.strftime('%B_%Y')
            id_str = f"{date_str}_USA"
            path_str = f"Esquire_{id_str}"

        # Build full URL
        url = f"{self.base_url}{id_str}/{path_str}"

        # Add suffix based on file format
        if self.file_format == 'jp2.zip':
            if with_text:
                url += "_text_jp2.zip"
            else:
                url += "_jp2.zip"
        else:
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

        Esquire uses 'x' as volume placeholder in filenames.

        Args:
            date: Issue date
            volume: Volume number (replaced with 'x')
            number: Issue number
            with_text: Whether this is a text-layer PDF

        Returns:
            Filename for local storage

        Example:
            Returns: "2020-01-x-5.pdf"
        """
        # Esquire uses 'x' instead of volume number
        if self._is_in_biweekly_period(date):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = date.strftime('%Y-%m')

        filename = f"{date_str}-x-{number}"

        # Add extension based on file format
        if self.file_format == 'jp2.zip':
            if with_text:
                filename += "_text_jp2.zip"
            else:
                filename += "_jp2.zip"
        else:
            if with_text:
                filename += "_text.pdf"
            else:
                filename += ".pdf"

        return filename

    def get_days_between_issues(self, date: dt.date) -> int:
        """
        Get days between issues for the given date.

        Returns:
            14 for biweekly period, 31 for monthly periods
        """
        if self._is_in_biweekly_period(date):
            return 14
        else:
            return 31
