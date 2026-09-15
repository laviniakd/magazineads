"""
DEPRECATED: Use unified_scraper.py instead.
    python unified_scraper.py --magazine the_nation [options]
"""
import warnings
warnings.warn(
    "scrape_the_nation.py is deprecated. Use unified_scraper.py --magazine the_nation instead.",
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
from utils import check_if_already_downloaded

import wget


# each url is of the form: https://archive.org/details/sim_nation_[year]-[month]-[day]_[vol]_[issue].pdf
# see https://archive.org/details/sim_nation_1945-01-06_160_1
def build_nation_url(date, volume, number, with_text=False):
    """
    Build the URL for a specific issue of the Nation.
    
    :param date: The date of the issue in 'YYYY-MM-DD' format.
    :param volume: The volume number of the issue.
    :param number: The issue number.
    :param with_text: If True, return the URL for the PDF with text.
    :return: The constructed URL as a string.
    """
    base_url = "https://archive.org/download/sim_nation_"
    date_obj = dt.datetime.strptime(date, '%Y-%m-%d').date()
    date_str = date_obj.strftime('%Y-%m-%d') # ...just making sure
    # if date is > 1951-07-07, no number!
    id_str = f"{date_str}_{volume}_{number}"

    url = f"{base_url}{id_str}/sim_nation_{id_str}" # quirk of archive.org URLs
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
    parser.add_option('--first-volume', type=int, default=1,
                      help='First volume: default=%default')
    parser.add_option('--first-number', type=int, default=1,
                      help='First number: default=%default')
    parser.add_option('--last-date', type=str, default='1959-01-05',
                      help='Last date: default=%default')
    parser.add_option('--last-volume', type=int, default=1, 
                      help='Last volume: default=%default')
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
    first_volume = options.first_volume
    first_number = options.first_number
    last_date = options.last_date
    last_volume = options.last_volume
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
    print("Downloading The Nation issues from", first_date, "to", last_date)


    volume = first_volume
    number = first_number

    failure_counter = 0


    done = False
    while not done:
        date_str = date.strftime('%Y-%m-%d')
        url = build_nation_url(date_str, volume, number, with_text=with_text)
        if with_text:
            outfile = os.path.join(data_dir, '-'.join([date_str, str(volume), str(number)]) + '_text.pdf')
        else:
            outfile = os.path.join(data_dir, '-'.join([date_str, str(volume), str(number)]) + '.pdf')      

        if check_if_already_downloaded(data_dir, date_str, volume, number):
            print("Already downloaded:", outfile)
        else:
            print("Downloading:", url)
            success = download_file(url, outfile, max_tries=3, overwrite=True)

            if not success:
                # first, try incrementing volume and resetting number to 1
                temp_volume = volume + 1
                new_url = build_nation_url(date_str, temp_volume, 1, with_text=with_text)
                success = download_file(new_url, outfile, max_tries=3, overwrite=True)
                if success:
                    volume = temp_volume
                    number = 1

            if not success:
                print("Failed on", date_str, volume, number)
                failure_counter += 1

            time.sleep(pause)
        
        number += 1
        date += dt.timedelta(days=7)
        # Volume number increases in July and January
        #if date.year > current_year:
        #    volume += 1
        #    number = 1
        #elif date.month == 7 and current_month == 6:
        #    volume += 1
        #    number = 1

        print("Current date:", date, "Volume:", volume, "Number:", number)
        print("Max date:", last_date, "Max volume:", last_volume, "Max number:", last_number)

        if failure_counter > 10:
            print("Too many failed scrapes, stopping script.")
            done = True

        if date > last_date or volume > last_volume or number > last_number:
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
