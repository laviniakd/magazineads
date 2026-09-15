"""Shared utility functions."""

import datetime as dt
import os


def check_if_already_downloaded(data_dir, date, volume, number, with_text=False):
    """Check if a specific issue of a magazine has already been downloaded.

    Used by legacy scraping scripts that construct filenames from date/volume/number.

    Args:
        data_dir: The directory where the issues are stored.
        date: The date of the issue in 'YYYY-MM-DD' or 'YYYY-MM' format.
        volume: The volume number of the issue.
        number: The issue number.
        with_text: If True, check for the PDF with text.

    Returns:
        True if the issue is already downloaded, False otherwise.
    """
    try:
        date_str = dt.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        date_str = dt.datetime.strptime(date, '%Y-%m').strftime('%Y-%m')
    filename = f"{date_str}-{volume}-{number}"
    if with_text:
        filename += "_text.pdf"
    else:
        filename += ".pdf"

    return os.path.exists(os.path.join(data_dir, filename))
