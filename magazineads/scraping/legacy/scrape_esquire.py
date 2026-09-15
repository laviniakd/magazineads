"""
DEPRECATED: Use unified_scraper.py instead.
    python unified_scraper.py --magazine esquire [options]
"""
import warnings
warnings.warn(
    "scrape_esquire.py is deprecated. Use unified_scraper.py --magazine esquire instead.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import json
import time
import datetime as dt
from glob import glob
from optparse import OptionParser
from collections import Counter

import wget


def check_if_already_downloaded(data_dir, date, number, with_text=False):
    """
    Check if a specific issue of The Economist has already been downloaded.
    
    :param data_dir: The directory where the issues are stored.
    :param date: The date of the issue in 'YYYY-MM-DD' format.
    :param number: The issue number.
    :param with_text: If True, check for the PDF with text.
    :return: True if the issue is already downloaded, False otherwise.
    """
    try:
        date_str = dt.datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        date_str = dt.datetime.strptime(date, '%Y-%m').strftime('%Y-%m')
    filename = f"{date_str}-x-{number}"
    if with_text:
        filename += "_text.pdf"
    else:
        filename += ".pdf"
    
    return os.path.exists(os.path.join(data_dir, filename))


# new urls are of the form https://archive.org/download/Esquire_April_2015_USA/Esquire_April_2015_USA.pdf
# i.e., need to build urls that look like https://archive.org/download/Esquire_[month_string]_[year]_USA/Esquire_[month_string]_[year]_USA.pdf
# old urls are of the form https://archive.org/download/Esquire-Magazine-1934-01/Esquire-Magazine-1934-01.pdf
# i.e., need to build urls that look like https://archive.org/download/Esquire-Magazine-[year]-[month]/Esquire-Magazine-[year]-[month].pdf
def build_esquire_url(date, with_text=False, old_format=True):
    """
    Build the URL for a specific issue of Esquire.

    :param date: The date of the issue in 'YYYY-MM-DD' format.
    :param with_text: If True, return the URL for the PDF with text.
    :return: The constructed URL as a string.
    """
    base_url = "https://archive.org/download/Esquire"
    date_obj = dt.datetime.strptime(date, '%Y-%m-%d').date()
    not_in_biweekly_period = date_obj.year < 1978 or (date_obj.year == 1978 and date_obj.month < 4) or (date_obj.year == 1979 and date_obj.month > 7) or (date_obj.year > 1979)

    if old_format and not_in_biweekly_period:
        date_str = date_obj.strftime('%Y-%m')
        id_str = f"-Magazine-{date_str}"
        url = f"{base_url}{id_str}/Esquire{id_str}" # quirk of archive.org URLs

    elif old_format: # is like "Esquire-Magazine-1978-04-11"
        date_str = date_obj.strftime('%Y-%m-%d')
        id_str = f"-Magazine-{date_str}"
        url = f"{base_url}{id_str}/Esquire{id_str}"

    else:
        date_str = date_obj.strftime('%B_%Y')
        id_str = f"{date_str}_USA"

        url = f"{base_url}_{id_str}/Esquire_{id_str}" # quirk of archive.org URLs
    
    if with_text:
        url += "_text.pdf"
    else:
        url += ".pdf"
    return url


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/magazines/life/',
                      help='Data directory: default=%default')
    parser.add_option('--first-date', type=str, default='1936-11-23',
                      help='First date: default=%default')
    parser.add_option('--first-number', type=int, default=1,
                      help='First number: default=%default')
    parser.add_option('--last-date', type=str, default='1959-01-05',
                      help='Last date: default=%default')
    parser.add_option('--last-number', type=int, default=1,
                        help='Last number: default=%default')
    parser.add_option('--pause', type=int, default=5,
                      help='Pause between issues: default=%default')
    parser.add_option('--with-text', action="store_true", default=False,
                      help='Get the PDFs with text (may be lower quality): default=%default')    
    #parser.add_option('--overwrite', action="store_true", default=False,
    #                  help='Overwrite files: default=%default')
    #parser.add_option('--clear-log', action="store_true", default=False,
    #                  help='Clear log file before starting: default=%default')

    (options, args) = parser.parse_args()

    data_dir = options.basedir
    first_date = options.first_date
    first_number = options.first_number
    last_date = options.last_date
    last_number = options.last_number
    pause = options.pause
    with_text = options.with_text

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    parts = first_date.split('-')
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    first_date = dt.date(year, month, day)
    date = first_date
    parts = last_date.split('-')
    year = int(parts[0])
    month = int(parts[1])
    day = int(parts[2])
    last_date = dt.date(year, month, day)
    print("Downloading Esquire issues from", first_date, "to", last_date)

    number = first_number

    failure_counter = 0


    done = False
    while not done:
        date_str = date.strftime('%Y-%m-%d')
        url = build_esquire_url(date_str, with_text=with_text)
        if with_text:
            outfile = os.path.join(data_dir, '-'.join([date_str, 'x', str(number)]) + '_text.pdf')
        else:
            outfile = os.path.join(data_dir, '-'.join([date_str, 'x', str(number)]) + '.pdf')

        if check_if_already_downloaded(data_dir, date_str, number):
            print("Already downloaded:", outfile)
        else:
            print("Downloading:", url)
            success = download_file(url, outfile, max_tries=3, overwrite=True)

            if not success:
                new_url = build_esquire_url(date_str, with_text=with_text)
                success = download_file(new_url, outfile, max_tries=3, overwrite=True)
                if success:
                    number = 1

            if not success:
                print("Failed on", date_str, number)
                failure_counter += 1

            time.sleep(pause)
        
        not_in_biweekly_period = date.year < 1978 or (date.year == 1978 and date.month < 4) or (date.year == 1979 and date.month > 7) or (date.year > 1979)
        if not_in_biweekly_period:
            number += 1
            date += dt.timedelta(days=31)
            date = date.replace(day=1)
        else: # were publishing biweekly during this period
            number += 1
            date += dt.timedelta(days=14)


        print("Current date:", date, "Number:", number)
        print("Max date:", last_date, "Max number:", last_number)

        if failure_counter > 10:
            print("Too many failed scrapes, stopping script.")
            done = True

        if date > last_date:
            done = True


def download_file(url, outfile, max_tries=3, logfile=None, overwrite=False):
    PAUSE_LENGTH = 0.5

    tries = 0
    success = False
    if os.path.exists(outfile) and not overwrite:
        print("Skipping {:s}".format(url))
    else:
        if os.path.exists(outfile) and overwrite:
            print("Deleting", outfile)
            os.remove(outfile)
        while tries < max_tries and not success:
            try:
                print("Downloading {:s}".format(url))
                wget.download(url, out=outfile)
                success = True
            except Exception as e:
                print("Download failed on", url)
                if os.path.exists(outfile):
                    os.remove(outfile)
                if tries < max_tries:
                    print(f"Pausing for {PAUSE_LENGTH} seconds...")
                    time.sleep(PAUSE_LENGTH)
                    tries += 1
                else:
                    print("Maximum number of tries exceeded on", url)
                    if logfile is not None:
                        with open(logfile, 'a') as f:
                            f.write("Maximum number of tries exceeded on" + url + '\n')
                    raise e
    print()
    return success


if __name__ == '__main__':
    main()
