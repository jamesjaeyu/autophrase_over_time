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

# https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip
#  => tmp/dblp.v10.zip
#  => tmp/dblp-ref/dblp-ref-0.json
#     tmp/dblp-ref/dblp-ref-1.json
#     tmp/dblp-ref/dblp-ref-2.json
#     tmp/dblp-ref/dblp-ref-3.json
#  =>
#------------------------------------
# data/dblp-v10/dblp-ref-0.json
# data/dblp-v10/dblp-ref-1.json
# data/dblp-v10/dblp-ref-2.json
# data/dblp-v10/dblp-ref-3.json
#------------------------------------
#config/data-params.json
#{
#    "infolder": "data/dblp-v10",
#    "dblp-v10-url": "https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip",
#    "dblp_v10_per_year_data_folder": "data/dblp-v10/txt",
#    "dblp_v10_paper_count_per_year_file": "data/dblp-v10/dblp-v10-counts.csv"
#}
def download_dblp_v10_using_requests(dblp_v10_url):
    dblp_v10_zip_file = "tmp/dblp.v10.zip"
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if os.path.exists('tmp/dblp-ref'):
        shutil.rmtree('tmp/dblp-ref')
    
    if os.path.exists(dblp_v10_zip_file):
        os.remove(dblp_v10_zip_file)
    
    # download and unzip the dataset
    print(f"Downloading dblp.v10.zip from {dblp_v10_url} ...")
    with requests.get(dblp_v10_url, stream=True) as resp:
        if not resp.ok:
            os.remove(dblp_v10_zip_file)
            sys.exit("Failed to dblp.v10.zip from {dblp_v10_url}")
        resp.raise_for_status()
        with open(dblp_v10_zip_file, 'wb') as zip_file:
            for chunk in resp.iter_content(chunk_size=8192): 
                zip_file.write(chunk)

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

    # num_issues:      total number of the mal-json-formated papers
    # num_no_abstract: total number of the papers with empty or no abstract and title
    # num_papers:      total number of papers
    # num_year:        total number of the valid papers but its year is out of the interested range [START_YEAR, END_YEAR] (year < START_YEAR or year > 2018)

    # 85.978 seconds has elapsed since start of function
    # 3079007 papers processed
    # 0 number of papers with json formatting issues (0.00%)
    # 530408 number of papers with empty abstracts (0.17%)
    # 82 number of papers with out of range/irrelevant years [1950, 2018] (0.00%)

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
                    #{
                    #  "abstract": "The purpose of this study is to develop a learning tool ... the basic principles of network protocols.",
                    #  "authors": [
                    #    "Makoto Satoh",
                    #    "Ryo Muramatsu",
                    #    "Mizue Kayama",
                    #    "Kazunori Itoh",
                    #    "Masami Hashimoto",
                    #    "Makoto Otani",
                    #    "Michio Shimizu",
                    #    "Masahiko Sugimoto"
                    #  ],
                    #  "n_citation": 0,
                    #  "references": [
                    #    "51c7e02e-f5ed-431a-8cf5-f761f266d4be",
                    #    "69b625b9-ebc5-4b60-b385-8a07945f5de9"
                    #  ],
                    #  "title": "Preliminary Design of a Network Protocol Learning Tool ... Design by an Empirical Study Using a Simple Mind Map",
                    #  "venue": "international conference on human-computer interaction",
                    #  "year": 2013,
                    #  "id": "00127ee2-cb05-48ce-bc49-9de556b93346"
                    #}
                    # ==>
                    # abstract\n
                    # title\n
                    #
                    # In this paper we propose multimodal extensions ... compared to state-of-the-art subspace clustering methods\n
                    # Multimodal Sparse and Low-Rank Subspace Clustering\n
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


def process_v10_txt(infolder):
    """
    Processes DBLP v10 dataset
    Outputs aggregate .txt files by year (only titles + abstracts)
    Takes around 8-9 minutes to process all papers
    >>> process_v10_txt('../data/dblp-v10')
    """
    start = time.time()
    filepaths = glob(infolder + '/*.json')
    outfolder = infolder + '/txt/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for fp in filepaths:
        file = open(fp)
        for line in file:
            try:
                data = json.loads(line)
                year = data['year']
                if year < 1950 or year > 2017:
                    continue
                outpath = outfolder + str(year) + '.txt'
                outfile = open(outpath, 'a')
                if 'abstract' in data.keys():
                    if len(data['abstract']) == 0:
                        continue
                    abstract = data['abstract']
                    abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
                    outfile.write(abstract + '\n')
                else:
                    continue
                if 'title' in data.keys():
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    outfile.write(title + '\n')
                else:
                    continue
            except:
                continue
        file.close()


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
    #filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
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
                    #outfile.write(abstract + '\n')
                else:
                    num_no_abstract += 1
                    continue
                if 'title' in data.keys():
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    content += title
                    #outfile.write(title + '\n')
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


def process_v10_csv(infolder):
    """
    Outputs aggregate .csv files by year
    Columns are: Title, Abstract

    >>> process_v10_csv('../data/dblp-v10')
    """
    start = time.time()
    filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outfolder = infolder + '/csv/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    years = set()

    num_no_abstract, num_no_title, num_papers, num_issues, num_year = 0, 0, 0, 0, 0

    for fp in filepaths:
        file = open(fp)
        for line in file:
            num_papers += 1
            try:
                data = json.loads(line)
                year = data['year']
                if year < 1950 or year > 2017:
                    num_year += 1
                    continue
                outpath = outfolder + str(year) + '.csv'
                outfile = open(outpath, 'a')
                if year not in years:
                    outfile.write('Title,Abstract\n')
                if 'abstract' in data.keys() and 'title' in data.keys():
                    if len(data['abstract']) == 0:
                        num_no_abstract += 1
                        continue
                    abstract = data['abstract']
                    abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    outfile.write(title + ',' + abstract + '\n')
                else:
                    num_no_abstract += 1
                    continue
                years.add(year)
            except:
                num_issues += 1
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)


def process_v10_txt_agg(infolder):
    """
    Outputs aggregated .txt file for titles + abstracts across all years

    >>> process_v10_txt_agg('../data/dblp-v10')
    """
    start = time.time()
    filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outfolder = infolder + '/agg/'
    outpath = outfolder + 'content_FULL.txt'
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
                if year < 1950 or year > 2017:
                    num_year += 1
                    continue
                outfile = open(outpath, 'a')
                if 'abstract' in data.keys():
                    if len(data['abstract']) == 0:
                        num_no_abstract += 1
                        continue
                    abstract = data['abstract']
                    abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
                    outfile.write(abstract + '\n')
                else:
                    num_no_abstract += 1
                    continue
                if 'title' in data.keys():
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    outfile.write(title + '\n')
                else:
                    continue
            except:
                num_issues += 1
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)

# autophrase_in_folder=data/dblp-v10/txt_grouped
# autophrase_out_folder=results/autophrase/dblp-v10/grouped
def autophrase_one_dir(autophrase_in_folder, autophrase_out_folder):
    """
    input: data/dblp-v10/txt_grouped
        data/dblp-v10/txt_grouped/1950-1959.txt
        data/dblp-v10/txt_grouped/1960-1964.txt
        ...
        data/dblp-v10/txt_grouped/2015-2017.txt

        1950-1959.txt: <abstract>\n<title>\n, e.g.
            This paper describes the logical nature of computing machines ... discover the laws of a very simple universe
            Logic discovery and the foundations of computing machinery
    
    output: results/autophrase/dblp-v10/grouped
        results/autophrase/dblp-v10/grouped/1950-1959
        results/autophrase/dblp-v10/grouped/1960-1964
        ...
        results/autophrase/dblp-v10/grouped/2015-2017
    """
    if not os.path.exists(autophrase_in_folder):
        print(f"{autophrase_in_folder} does not exists.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(autophrase_out_folder):
        shutil.rmtree(autophrase_out_folder)
    os.makedirs(autophrase_out_folder, exist_ok=True)

    print(f'{autophrase_in_folder} => {autophrase_out_folder}')

    # iterate through the files (e.g. 1950-1959.txt) under "data/dblp-v10/txt_grouped"
    for filename in sorted(os.listdir(autophrase_in_folder)):
        # 1950-1959.txt => 1950-1959
        model_dir = autophrase_out_folder + '/' + filename.split('.')[0]
        # delete the results of previous run if there is any
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        # mkdir results/autophrase/dblp-v10/grouped/1950-1959
        os.makedirs(model_dir, exist_ok=True)

        one_year_file = os.path.join(autophrase_in_folder, filename)
        # one_year_file = data/dblp-v10/txt_grouped/1950-1959.txt
        # model_dir = results/autophrase/dblp-v10/grouped/1950-1959
        perform_autophrase(one_year_file, model_dir)

# input_file=data/dblp-v10/txt_grouped/1950-1959.txt
# model_dir=results/autophrase/dblp-v10/grouped/1950-1959
def perform_autophrase(input_file, model_dir):
    """
    input:  data/dblp-v10/txt_grouped/1950-1959.txt
    output: results/autophrase/dblp-v10/grouped/1950-1959
                AutoPhrase_multi-words.txt
                AutoPhrase_single-word.txt
                AutoPhrase.txt
                language.txt
                segmentation_extracted_phrases.csv
                segmentation_extracted_phrases.txt
                segmentation.model
                segmentation.txt
                token_mapping.txt    
    """
    MIN_SUP = 20
    MODEL = "../" + model_dir
    RAW_TRAIN = "../" + input_file
    TEXT_TO_SEG = RAW_TRAIN
    THREAD = 64

    autophrase_params = f"MIN_SUP={MIN_SUP} MODEL={MODEL} RAW_TRAIN={RAW_TRAIN} TEXT_TO_SEG={TEXT_TO_SEG} THREAD={THREAD}"

    # MIN_SUP=20 MODEL=../results/autophrase/dblp-v10/grouped/1950-1959 RAW_TRAIN=../data/dblp-v10/txt_grouped/1950-1959.txt TEXT_TO_SEG=../data/dblp-v10/txt_grouped/1950-1959.txt THREAD=1 ./auto_phrase.sh
    # auto_phrase.sh
    # input:
    #     data/dblp-v10/txt_grouped/1950-1959.txt
    # output:
    #     results/autophrase/dblp-v10/grouped/1950-1959/AutoPhrase_multi-words.txt
    #     results/autophrase/dblp-v10/grouped/1950-1959/AutoPhrase_single-word.txt
    #     results/autophrase/dblp-v10/grouped/1950-1959/AutoPhrase.txt
    #     results/autophrase/dblp-v10/grouped/1950-1959/language.txt
    #     results/autophrase/dblp-v10/grouped/1950-1959/segmentation.model
    #     results/autophrase/dblp-v10/grouped/1950-1959/token_mapping.txt
    #
    #     where AutoPhrase.txt = AutoPhrase_single-word.txt + AutoPhrase_multi-words.txt
    #     therefore, use AutoPhrase.txt is good enough

    # MIN_SUP=20 MODEL=../results/autophrase/dblp-v10/grouped/1950-1959 RAW_TRAIN=../data/dblp-v10/txt_grouped/1950-1959.txt TEXT_TO_SEG=../data/dblp-v10/txt_grouped/1950-1959.txt THREAD=1 ./phrasal_segmentation.sh
    # phrasal_segmentation.sh
    # input:
    #     data/dblp-v10/txt_grouped/1950-1959.txt
    # output:
    #     results/autophrase/dblp-v10/grouped/1950-1959/segmentation.txt
    #       This <phrase>paper</phrase> describes the logical <phrase>nature</phrase> of <phrase>computing</phrase> machines 
    #       in terms of languages and the types of problems that can be solved by logical operations on languages 
    #       The problem of discovery in <phrase>mathematics</phrase> and empirical <phrase>science</phrase> is discussed 
    #       and an inductive machine is described which would be able to formulate hypotheses modify them in the <phrase>light</phrase> 
    #       of new experience and eventually discover the laws of a very simple <phrase>universe</phrase>
    #       <phrase>Logic</phrase> discovery and the foundations of <phrase>computing</phrase> machinery
    cmd = f'cd AutoPhrase && {autophrase_params} ./auto_phrase.sh && {autophrase_params} ./phrasal_segmentation.sh'
    start = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: {cmd}", flush=True)
    os.system(cmd)
    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{current_datetime}: {input_file} => {model_dir} {(end - start):.03f} seconds', flush=True)