"""
Common download utilities for magazine scraping.

Extracted from individual scraping scripts to eliminate duplication.
"""

import os
import time
import wget
from typing import Optional

# Import check_if_already_downloaded from magazineads.utils to avoid duplication
from magazineads.utils import check_if_already_downloaded


def download_file(
    url: str,
    outfile: str,
    max_tries: int = 3,
    pause_length: float = 0.5,
    logfile: Optional[str] = None,
    overwrite: bool = False
) -> bool:
    """
    Download a file from a URL with retry logic.

    Extracted from individual scraping scripts (scrape_variety.py, scrape_esquire.py, etc.)
    which all had identical or near-identical implementations.

    Args:
        url: URL to download from
        outfile: Path to save the downloaded file
        max_tries: Maximum number of download attempts (default: 3)
        pause_length: Seconds to wait between retry attempts (default: 0.5)
                     Note: Some magazines use 3 seconds, configurable per magazine
        logfile: Optional file to log failures to
        overwrite: Whether to overwrite existing files (default: False)

    Returns:
        bool: True if download succeeded, False if file was skipped (already exists)

    Raises:
        Exception: If maximum retries exceeded
    """
    tries = 0
    success = False

    if os.path.exists(outfile) and not overwrite:
        print(f"Skipping {url}")
        return False

    if os.path.exists(outfile) and overwrite:
        print(f"Deleting {outfile}")
        os.remove(outfile)

    while tries < max_tries and not success:
        try:
            print(f"Downloading {url}")
            wget.download(url, out=outfile)
            success = True
        except Exception as e:
            print(f"Download failed on {url}")
            if os.path.exists(outfile):
                os.remove(outfile)

            tries += 1
            if tries < max_tries:
                print(f"Pausing for {pause_length} seconds...")
                time.sleep(pause_length)
            else:
                print(f"Maximum number of tries exceeded on {url}")
                if logfile is not None:
                    with open(logfile, 'a') as f:
                        f.write(f"Maximum number of tries exceeded on {url}\n")
                raise e

    print()
    return success


# Re-export check_if_already_downloaded for convenience
__all__ = ['download_file', 'check_if_already_downloaded']
