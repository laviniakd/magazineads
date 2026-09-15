"""Legacy scrapers using template-based URL construction.

These scrapers construct Internet Archive download URLs by predicting the exact
identifier from date/volume/number patterns. This approach is fragile — it
requires knowing each magazine's volume numbering scheme and breaks when IA
identifiers don't follow the expected pattern.

For new downloads, use ``magazineads.scraping.ia_scraper`` instead, which
queries the Internet Archive search API to discover items dynamically.

These scripts are preserved because they were used to build the original corpus
described in the dataset paper.
"""
