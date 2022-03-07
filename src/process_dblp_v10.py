"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import json
import datetime
import time
import sys
import re
import os
import requests
import shutil
import zipfile
from glob import glob

START_YEAR = 1950
END_YEAR = 2018

year_filename_mapping = {
    1950: '1950-1959',
    1951: '1950-1959',
    1952: '1950-1959',
    1953: '1950-1959',
    1954: '1950-1959',
    1955: '1950-1959',
    1956: '1950-1959',
    1957: '1950-1959',
    1958: '1950-1959',
    1959: '1950-1959',
    1960: '1960-1964',
    1961: '1960-1964',
    1962: '1960-1964',
    1963: '1960-1964',
    1964: '1960-1964',
    1965: '1965-1969',
    1966: '1965-1969',
    1967: '1965-1969',
    1968: '1965-1969',
    1969: '1965-1969',
    1970: '1970-1974',
    1971: '1970-1974',
    1972: '1970-1974',
    1973: '1970-1974',
    1974: '1970-1974',
    1975: '1975-1979',
    1976: '1975-1979',
    1977: '1975-1979',
    1978: '1975-1979',
    1979: '1975-1979',
    1980: '1980-1984',
    1981: '1980-1984',
    1982: '1980-1984',
    1983: '1980-1984',
    1984: '1980-1984',
    1985: '1985-1989',
    1986: '1985-1989',
    1987: '1985-1989',
    1988: '1985-1989',
    1989: '1985-1989',
    1990: '1990-1994',
    1991: '1990-1994',
    1992: '1990-1994',
    1993: '1990-1994',
    1994: '1990-1994',
    1995: '1995-1999',
    1996: '1995-1999',
    1997: '1995-1999',
    1998: '1995-1999',
    1999: '1995-1999',
    2000: '2000-2004',
    2001: '2000-2004',
    2002: '2000-2004',
    2003: '2000-2004',
    2004: '2000-2004',
    2005: '2005-2009',
    2006: '2005-2009',
    2007: '2005-2009',
    2008: '2005-2009',
    2009: '2005-2009',
    2010: '2010-2014',
    2011: '2010-2014',
    2012: '2010-2014',
    2013: '2010-2014',
    2014: '2010-2014',
    2015: '2015-2017',
    2016: '2015-2017',
    2017: '2015-2017',
    2018: '2015-2017',
    2019: '2015-2017'}

def year_to_filename(year, year_grouping = False, output_type = 'txt'):
    if year_grouping:
        # look for year_filename_mapping for mapping:
        # 1950 => 1950-1959.txt or 1950-1959.csv
        return f'{year_filename_mapping[year]}.{output_type}'
    # not grouping: year.txt/year.csv
    #    1950 => 1950.txt
    #    1950 => 1950.csv
    return f'{year}.{output_type}'

# common phrases to be excluded
stop_phrase_list = [
    "paper", "abstractthis paper", "abstractthis paper investigates", "analysis of",
    "cited papers", "companion paper",
    "invited paper", "paper addresses", "paper analyzes",
    "paper argues", "paper compares", "paper considers", "paper describes",
    "paper discusses", "paper documents", "paper ends", "paper evaluates",
    "paper examines", "paper explains", "paper explores", "paper illustrates",
    "paper introduces", "paper investigates", "paper outlines",
    "paper presents", "paper proposes", "paper puts forward", "paper reports",
    "paper reviews", "paper revisits", "paper shows", "papers published",
    "paper summarizes", "previous papers", "published papers",
    "research papers", "scientific papers", "selected papers", "seminal paper",
    "short paper", "survey paper", "a case study", "worst case", "worst-case", "write",
    "the problem", "this paper", "by"]

def skip_phrase(phrase):
    """
    exclude common phrases and single-character-phrase
    """
    if len(phrase) <= 1:
        return True
    if phrase in stop_phrase_list:
        return True
    return False

year_bracket_int_mapping = {
    '1950-1959': 0,
    '1960-1964': 1,
    '1965-1969': 2,
    '1970-1974': 3,
    '1975-1979': 4,
    '1980-1984': 5,
    '1985-1989': 6,
    '1990-1994': 7,
    '1995-1999': 8,
    '2000-2004': 9,
    '2005-2009': 10,
    '2010-2014': 11,
    '2015-2017': 12
}

def year_bracket_to_int(year_bracket):
    encoded_int = year_bracket_int_mapping.get(year_bracket)
    if encoded_int is None:
        print(f"{year_bracket} unknown.", file=sys.stderr)
    return encoded_int


def download_dblp_v10(dblp_v10_url):
    """
    Downloads DBLP v10 dataset and creates directory
    """
    dblp_v10_zip_file = "tmp/dblp.v10.zip"
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if os.path.exists('tmp/dblp-ref'):
        shutil.rmtree('tmp/dblp-ref')

    if os.path.exists(dblp_v10_zip_file):
        os.remove(dblp_v10_zip_file)
    os.system(f"wget -O {dblp_v10_zip_file} {dblp_v10_url}")

    with zipfile.ZipFile(dblp_v10_zip_file, 'r') as zip_ref:
        zip_ref.extractall("tmp")
    
    filepaths = [f'dblp-ref-{num}.json' for num in range(4)]
    for file in filepaths:
        print(f"mv tmp/dblp-ref/{file} data/dblp-v10/{file}")
        shutil.move(f"tmp/dblp-ref/{file}", f"data/dblp-v10/{file}")
    shutil.rmtree('tmp/dblp-ref')


def print_stats(time, papers, issues, empty, year):
    """
    159.422 seconds has elapsed since start of function
    3079007 papers processed
    0 number of papers with json formatting issues (0.00%)
    530408 number of papers with empty abstracts (0.17%)
    82 number of papers with out of range/irrelevant years [1950, 2018] (0.00%)
    """
    print(f'{time:.3f} seconds has elapsed since start of function')
    print(f'{papers} papers processed')
    print(f'{issues} number of papers with json formatting issues ({(issues/papers):.2f}%)')
    print(f'{empty} number of papers with empty abstracts ({(empty/papers):.2f}%)')
    print(f'{year} number of papers with out of range/irrelevant years [{START_YEAR}, {END_YEAR}] ({(year/papers):.2f}%)')


def process_v10(infolder, outfolder, year_grouping = False, output_type = 'txt'):
    """
    Processes DBLP v10 dataset
    Outputs aggregate .txt files by year (only titles + abstracts)

    Takes around 8-9 minutes to process all papers

    run.py: process_v10_txt(data_cfg['infolder'], data_cfg['outfolder'], want_mapping)

    Input:
        data/dblp-v10/dblp-ref-0.json
        data/dblp-v10/dblp-ref-1.json
        data/dblp-v10/dblp-ref-2.json
        data/dblp-v10/dblp-ref-3.json

    Output:
        data/dblp-v10/txt/1950.txt
        data/dblp-v10/txt/1951.txt
        ...
        data/dblp-v10/txt/2018.txt
    Format of data/dblp-v10/txt/YYYY.txt
        abstract\n
        title\n
        ...
    """
    start = time.time()
    print(f'infolder:{infolder} outfolder:{outfolder} year_grouping:{year_grouping} output_type:{output_type}')
    num_no_abstract, num_papers, num_issues, num_year = 0, 0, 0, 0

    # cache opened file handlers so that we do not need to open and close file for every paper
    # for better performance.  100% speed improvement.
    # year => the file handler to data/dblp-v10/txt/YYYY.txt
    file_handlers = {}

    # need to delete the existing resultant files of the previous run.
    # otherwise, the results of different runs will be appended together
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.mkdir(outfolder)

    abstract_title_separator = '\n'
    if output_type == 'csv':
        abstract_title_separator = ','

    # input files
    # 0 1 2 3
    filepaths = [f'{infolder}/dblp-ref-{num}.json' for num in range(4)]
    for fp in filepaths:
        print(f'processing {fp}')
        # handle file open error
        try:
            file = open(fp)
        except OSError as ex:
            print(f"{ex}", file=sys.stderr)
            continue
        
        # use with statement and opened file handler will be closed automatically
        # https://www.geeksforgeeks.org/with-statement-in-python
        with file:
            for line in file:
                num_papers += 1
                try:
                    data = json.loads(line)
                    # use get() of the dictionary.  If key does not exist, get() will return None and no need to check if key exists or not
                    year = data.get('year')
                    if year is None or year < START_YEAR or year > END_YEAR:
                        num_year += 1
                        continue
                    else:
                        year = data['year']
                    # if a file of YYYY.txt has been opened before, use the cached file handler.
                    # year => the file handler to data/dblp-v10/txt/YYYY.txt
                    outfile = file_handlers.get(year)
                    if outfile is None:
                        # it is the first time for year, open file and cache it to file_handlers
                        outpath = f'{outfolder}/{year_to_filename(year, year_grouping, output_type)}'
                        try:
                            outfile = open(outpath, 'a')
                            file_handlers[year] = outfile
                        except OSError as ex:
                            print(f"{ex}", file=sys.stderr)
                            # close all of the opened file handlers
                            for file_year in file_handlers.keys():
                                file_handlers[year].close()
                            return
                    abstract = data.get('abstract')
                    title = data.get('title')
                    # check either of abstract is None or empty string
                    # instead of "abstract is None or title is None:" (only check for None)
                    if not abstract:
                        num_no_abstract += 1
                        continue

                    abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)

                    # skip the paper with empty abstract or title after excluding the non-alpha-numeric characters
                    if not abstract:
                        num_no_abstract += 1
                    elif not title:
                        outfile.write(abstract)
                        outfile.write('\n')
                    else:
                        outfile.write(abstract)
                        outfile.write(abstract_title_separator)
                        outfile.write(title)
                        outfile.write('\n')
                except:
                    num_issues += 1
    # close all of the opened file handlers to avoid file descriptor leak
    for file_year in file_handlers.keys():
        file_handlers[year].close()
    end = time.time()
    total_time_used = end - start
    print_stats(total_time_used, num_papers, num_issues, num_no_abstract, num_year)
    print(f'{START_YEAR}-{END_YEAR}: input:{infolder} output:{outfolder} {total_time_used:.3f} seconds', flush=True)


def process_v10_txt_grouped(infolder):
    """
    Processes DBLP v10 but with years grouped by 5

    >>> process_v10_txt_grouped('../data/dblp-v10')
    """
    start = time.time()
    # Creates dictionary mapping for each year's new year-range category
    mapping = {}
    for i in range(1950, 2018, 5):
        if i == 1950 or i == 1955:
            cat = '1950-1959'
        elif i == 2015:
            cat = '2015-2017'
        else:
            cat = str(i) + '-' + str(i+4)
        for j in range(i, i+5):
            mapping[j] = cat

    # Processes input files
    filepaths = glob(infolder + '/*.json')
    outfolder = infolder + '/txt_grouped/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    num_no_abstract, num_papers, num_issues, num_year = 0, 0, 0, 0
    for fp in filepaths:
        file = open(fp)
        for line in file:
            num_papers += 1
            try:
                data = json.loads(line)
                year = data['year']
                if year < 1950 or year > 2018:
                    num_year += 1
                    continue
                year = mapping[year]
                outpath = outfolder + str(year) + '.txt'
                outfile = open(outpath, 'a')
                content = ''
                if 'abstract' in data.keys():
                    if len(data['abstract']) == 0:
                        num_no_abstract += 1
                        continue
                    abstract = data['abstract']
                    abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
                    content += abstract + ' '
                    outfile.write(abstract + '\n')
                else:
                    num_no_abstract += 1
                    continue
                if 'title' in data.keys():
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    content += title
                    outfile.write(title + '\n')
                if content != '':
                    outfile.write(content + '\n')
                else:
                    continue
            except:
                num_issues += 1
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)


# common phrases to be excluded
stop_phrase_list = [
    "abstractthis paper", "abstractthis paper investigates", "analysis of",
    "cited papers", "companion paper",
    "invited paper", "paper addresses", "paper analyzes",
    "paper argues", "paper compares", "paper considers", "paper describes",
    "paper discusses", "paper documents", "paper ends", "paper evaluates",
    "paper examines", "paper explains", "paper explores", "paper illustrates",
    "paper introduces", "paper investigates", "paper outlines",
    "paper presents", "paper proposes", "paper puts forward", "paper reports",
    "paper reviews", "paper revisits", "paper shows", "papers published",
    "paper summarizes", "previous papers", "published papers",
    "research papers", "scientific papers", "selected papers", "seminal paper",
    "short paper", "survey paper", "a case study", "worst case", "worst-case", "write",
    "the problem", "this paper"]

def skip_phrase(phrase):
    """
    exclude common phrases and single-character-phrase
    """
    if len(phrase) <= 1:
        return True
    if phrase in stop_phrase_list:
        return True
    return False

# autophrase_out_folder=results/autophrase/dblp-v10/grouped
def extract_phrases_one_dir(autophrase_out_folder):
    """
    input: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959/segmentation.txt
        results/autophrase/dblp-v10/grouped/1960-1965/segmentation.txt
        ...
        results/autophrase/dblp-v10/grouped/2015-2017/segmentation.txt
            This <phrase>paper</phrase> describes the logical <phrase>nature</phrase> of <phrase>computing</phrase> machines ..
            problem of discovery in <phrase>mathematics</phrase> and empirical <phrase>science</phrase> is discussed ...
            formulate hypotheses modify them in the <phrase>light</phrase> of new experience and eventually ...
            simple <phrase>universe</phrase>
            <phrase>Logic</phrase> discovery and the foundations of <phrase>computing</phrase> machinery

    output: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959/segmentation_extracted_phrases.txt
        results/autophrase/dblp-v10/grouped/1960-1965/segmentation_extracted_phrases.txt
        ...
        results/autophrase/dblp-v10/grouped/2015-2017/segmentation_extracted_phrases.txt
            paper
            nature
            computing
            mathematics
            ...
    """
    if not os.path.exists(autophrase_out_folder):
        print(f"{autophrase_out_folder} does not exists.", file=sys.stderr)
        sys.exit(1)

    # compiled regex for better speed
    phrase_regex = re.compile(r"<phrase>(.*?)</phrase>", flags=0)

    for dirname in sorted(os.listdir(autophrase_out_folder)):
        a_dir = os.path.join(autophrase_out_folder, dirname)
        # results/autophrase/dblp-v10/grouped/1950-1959
        if not os.path.isdir(a_dir):
            continue
        seg_filename = os.path.join(a_dir, "segmentation.txt")
        if not os.path.exists(seg_filename):
            continue
        
        extracted_phrases_filename = os.path.join(a_dir, "segmentation_extracted_phrases.txt")
        if os.path.exists(extracted_phrases_filename):
            os.remove(extracted_phrases_filename)

        start = time.time()
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_datetime}: {seg_filename} => {extracted_phrases_filename} ...")
        try:
            extracted_phrases_file = open(extracted_phrases_filename, 'a')
        except OSError as ex:
            print(f"{ex}", file=sys.stderr)
            sys.exit(1)
        with extracted_phrases_file:
            with open(seg_filename, "r") as seg_file:
                for line in seg_file:
                    for phrase in phrase_regex.findall(line):
                        lower_phrase = phrase.lower()
                        if skip_phrase(lower_phrase):
                            continue
                        extracted_phrases_file.write(lower_phrase)
                        extracted_phrases_file.write('\n')
        end = time.time()
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_datetime}: {seg_filename} => {extracted_phrases_filename} {(end - start):.03f} seconds", flush=True)


# cat segmentation_extracted_phrases.txt  | sort | uniq -c | sort -k1nr 2>&1 | sed -e 's/^[ ]*//g' | sed 's/^\([[:digit:]]*\) /\1,/g'
# autophrase_out_folder=results/autophrase/dblp-v10/grouped
def count_phrase_one_dir(autophrase_out_folder):
    """
    count and sort phrases based on their total counter (desc)
    We use Linux commands instead of pandas.Dataframe for performance.
    Make sure that we have cat, sort, uniq, and sed on a Windows PC
    cat segmentation_extracted_phrases.txt  | sort | uniq -c | sort -k1nr 2>&1 | sed -e 's/^[ ]*//g' | sed 's/^\([[:digit:]]*\) /\1,/g'

    input: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959/segmentation_extracted_phrases.txt
        results/autophrase/dblp-v10/grouped/1960-1965/segmentation_extracted_phrases.txt
        ...
        results/autophrase/dblp-v10/grouped/2015-2017/segmentation_extracted_phrases.txt
            paper
            nature
            computing
            mathematics
            ...

    output: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959/segmentation_extracted_phrase_count.csv
        results/autophrase/dblp-v10/grouped/1960-1964/segmentation_extracted_phrase_count.csv
        ...
        results/autophrase/dblp-v10/grouped/2015-2017/segmentation_extracted_phrase_count.csv
            125,digital
            107,paper
            91,information
            87,function
    """
    if not os.path.exists(autophrase_out_folder):
        print(f"{autophrase_out_folder} does not exists.", file=sys.stderr)
        sys.exit(1)

    for dirname in sorted(os.listdir(autophrase_out_folder)):
        a_dir = os.path.join(autophrase_out_folder, dirname)
        # results/autophrase/dblp-v10/grouped/1950-1959
        if not os.path.isdir(a_dir):
            continue
        segmentation_extracted_phrases_txt_filename = os.path.join(a_dir, "segmentation_extracted_phrases.txt")
        if not os.path.exists(segmentation_extracted_phrases_txt_filename):
            print(f"{segmentation_extracted_phrases_txt_filename} does not exists.", file=sys.stderr)
            continue
        
        extracted_phrases_count_csv_filename = os.path.join(a_dir, "segmentation_extracted_phrase_count.csv")
        cmd = f"cat {segmentation_extracted_phrases_txt_filename}  | sort | uniq -c | sort -k1nr | sed -e 's/^[ ]*//g' | sed 's/^\([[:digit:]]*\) /\\1,/g' > {extracted_phrases_count_csv_filename}"

        start = time.time()
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_datetime}: {segmentation_extracted_phrases_txt_filename} => {extracted_phrases_count_csv_filename} ...")
        os.system(cmd)
        end = time.time()
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{current_datetime}: {segmentation_extracted_phrases_txt_filename} => {extracted_phrases_count_csv_filename} {(end - start):.03f} seconds", flush=True)

# input: autophrase_dblp_v10_grouped_folder=results/autophrase/dblp-v10/grouped
# output: extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
def extract_phrases_per_paper_one_dir(autophrase_dblp_v10_grouped_folder, extract_phrases_per_paper_csv_file, only_first_year_bracket = False):
    """
    input: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959/segmentation.txt
        results/autophrase/dblp-v10/grouped/1960-1965/segmentation.txt
        ...
        results/autophrase/dblp-v10/grouped/2015-2017/segmentation.txt
            This <phrase>paper</phrase> describes the logical <phrase>nature</phrase> of <phrase>computing</phrase> machines ..
            problem of discovery in <phrase>mathematics</phrase> and empirical <phrase>science</phrase> is discussed ...
            formulate hypotheses modify them in the <phrase>light</phrase> of new experience and eventually ...
            simple <phrase>universe</phrase>
            <phrase>Logic</phrase> discovery and the foundations of <phrase>computing</phrase> machinery

    output: results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
        paper_id,"phrases",year_bracket,year_bracket_int_encoded
        1,"paper,nature",1950-1959,0
        ...
    """
    if not os.path.exists(autophrase_dblp_v10_grouped_folder):
        print(f"{autophrase_dblp_v10_grouped_folder} does not exists.", file=sys.stderr)
        sys.exit(1)

    # extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
    if os.path.exists(extract_phrases_per_paper_csv_file):
        os.remove(extract_phrases_per_paper_csv_file)

    try:
        extract_phrases_per_paper_csv_file_fp = open(extract_phrases_per_paper_csv_file, 'a')
    except OSError as ex:
        print(f"{ex}", file=sys.stderr)
        sys.exit(1)

    # compiled regex for better speed
    phrase_regex = re.compile(r"<phrase>(.*?)</phrase>", flags=0)

    paper_id = 0
    with extract_phrases_per_paper_csv_file_fp:
        for dirname in sorted((f for f in os.listdir(autophrase_dblp_v10_grouped_folder) if not f.endswith(".pkl") and not f.endswith(".csv")), key=str.lower):
            a_dir = os.path.join(autophrase_dblp_v10_grouped_folder, dirname)
            # results/autophrase/dblp-v10/grouped/1950-1959
            if not os.path.isdir(a_dir):
                continue
            seg_filename = os.path.join(a_dir, "segmentation.txt")
            if not os.path.exists(seg_filename):
                continue
            
            # results/autophrase/dblp-v10/grouped/1950-1959 => 1950-1959
            year_bracket = a_dir.split('\\')[-1]

            start = time.time()
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{current_datetime}: {seg_filename} => {extract_phrases_per_paper_csv_file} ...", end=" ")
            with open(seg_filename, "r") as seg_file:
                per_file_line_counter = 0
                paper_phrases = []
                for line in seg_file:
                    for phrase in phrase_regex.findall(line):
                        lower_phrase = phrase.lower()
                        if not skip_phrase(lower_phrase):
                            paper_phrases.append(lower_phrase)
                    # each paper has two lines: abstract and title
                    if per_file_line_counter % 2 == 1:
                        phrases_csv = '_'.join(paper_phrases)
                        #print(f"{paper_phrases} {phrases_csv}")
                        if len(phrases_csv) > 0:
                            year_bracket_int_encoded = year_bracket_to_int(year_bracket)
                            # 1,transistors_semiconductor_transistor_amplifiers_binary_cps,1950-1959,0
                            content = f'{paper_id},{phrases_csv},{year_bracket},{year_bracket_int_encoded}'
                            extract_phrases_per_paper_csv_file_fp.write(content)
                            extract_phrases_per_paper_csv_file_fp.write('\n')
                        else:
                            print(f"empty:phrases_csv: {line}")
                        paper_id += 1
                        paper_phrases = []
                    per_file_line_counter += 1
            end = time.time()
            print(f"{(end - start):.03f} seconds", flush=True)
            if only_first_year_bracket:
                return

# input: extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
# output: extract_phrases_per_paper_pkl_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl
def convert_extract_phrases_per_paper_csv_to_df_pickle(extract_phrases_per_paper_csv_file, extract_phrases_per_paper_pkl_file):
    """
        input: extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
            0,nature_computing_mathematics_science_light_universe_logic_computing,1950-1959,0
            1,transistors_semiconductor_transistor_amplifiers_binary_cps,1950-1959,0
        output: extract_phrases_per_paper_pkl_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl
            replace "_" with " "
            paper_id, phrases, year_bracket, year_bracket_int_encoded
            0,nature computing mathematics science light universe logic computing,1950-1959,0
            1,transistors semiconductor transistor amplifiers binary cps,1950-1959,0
    """
    if not os.path.exists(extract_phrases_per_paper_csv_file):
        print(f"{extract_phrases_per_paper_csv_file} does not exists.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(extract_phrases_per_paper_pkl_file):
        os.remove(extract_phrases_per_paper_pkl_file)

    df = pd.read_csv(extract_phrases_per_paper_csv_file, names=['paper_id', 'phrases', 'year_bracket', 'year_bracket_int_encoded'])
    df['phrases'] = df.apply(lambda x: x['phrases'].replace("_", " ") if isinstance(x['phrases'], str) else np.nan, axis=1)
    df.to_pickle(extract_phrases_per_paper_pkl_file)
