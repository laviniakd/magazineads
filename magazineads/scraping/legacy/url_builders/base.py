"""
Template-based URL builder for magazines with simple URL patterns.

Handles magazines where URLs follow a consistent template pattern,
differing only in magazine name and date/volume/number formatting.
"""

import datetime as dt
from typing import Dict, Any


class TemplateURLBuilder:
    """
    Build download URLs using string template substitution.

    Suitable for magazines with straightforward URL patterns like:
    - Variety: https://archive.org/download/sim_variety_{date}_{volume}_{number}/sim_variety_{date}_{volume}_{number}.pdf
    - The Nation: https://archive.org/download/sim_nation_{date}_{volume}_{number}/sim_nation_{date}_{volume}_{number}.pdf

    URL pattern is defined in magazine configuration JSON.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize URL builder with magazine configuration.

        Args:
            config: Magazine configuration dictionary containing:
                - base_url: Base URL template
                - url_pattern: Pattern for issue identifier
                - url_path_pattern: Pattern for file path
                - date_format: strftime format for dates
                - name: Magazine name for substitution
        """
        self.config = config
        self.base_url = config['base_url']
        self.url_pattern = config['url_pattern']
        self.url_path_pattern = config['url_path_pattern']
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
        Build download URL for a specific magazine issue.

        Args:
            date: Issue date
            volume: Volume number
            number: Issue number
            with_text: Whether to get PDF with text layer

        Returns:
            Complete download URL

        Example:
            For Variety with date 2020-01-15, volume 337, number 1:
            Returns: https://archive.org/download/sim_variety_2020-01-15_337_1/sim_variety_2020-01-15_337_1.pdf
        """
        # Format date according to magazine's date format
        date_str = date.strftime(self.date_format)

        # Build issue identifier using pattern
        issue_id = self.url_pattern.format(
            date=date_str,
            volume=volume,
            number=number,
            magazine=self.magazine_name
        )

        # Build file path using pattern
        file_path = self.url_path_pattern.format(
            date=date_str,
            volume=volume,
            number=number,
            magazine=self.magazine_name
        )

        # Combine into full URL
        url = f"{self.base_url}{issue_id}/{file_path}"

        # Add appropriate suffix
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
        Build local filename for downloaded magazine issue.

        Args:
            date: Issue date
            volume: Volume number
            number: Issue number
            with_text: Whether this is a text-layer PDF

        Returns:
            Filename for local storage

        Example:
            Returns: "2020-01-15-337-1.pdf" or "2020-01-15-337-1_text.pdf"
        """
        date_str = date.strftime(self.date_format)
        filename_pattern = self.config.get('filename_pattern', '{date}-{volume}-{number}')

        filename = filename_pattern.format(
            date=date_str,
            volume=volume,
            number=number
        )

        if with_text:
            filename += "_text.pdf"
        else:
            filename += ".pdf"

        return filename
