"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import json
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
