"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import json
import time
import re
import os
import requests
import shutil
from glob import glob

def download_v10():
    """
    Downloads DBLP v10 dataset and creates directory
    """
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/dblp-v10'):
        os.mkdir('data/dblp-v10')

    dl_link = 'https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip'

    data = open('data/dblp-v10/dblp-v10.zip', 'wb')
    r = requests.get(dl_link, allow_redirects=True)
    data.write(r.content)
    shutil.unpack_archive('data/dblp-v10/dblp-v10.zip', 'data/dblp-v10')
    return


def print_stats(time, papers, issues, empty, year):
    print(str(time) + ' seconds has elapsed since start of function')
    print(str(papers) + ' papers processed')
    print(str(issues) + ' number of papers with json formatting issues (' + str(issues/papers)[:5] + ')')
    print(str(empty) + ' number of papers with empty abstracts (' + str(empty/papers)[:5] + ')')
    print(str(year) + ' number of papers with invalid/irrelevant years < 1950, > 2017')


def process_v10_txt(infolder):
    """
    Processes DBLP v10 dataset
    Outputs aggregate .txt files by year (only titles + abstracts)

    Takes around 8-9 minutes to process all papers

    >>> process_v10_txt('../data/dblp-v10')
    """
    start = time.time()
    filepaths = glob(infolder + '/*.json')

    #filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outfolder = infolder + '/txt/'
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
                outpath = outfolder + str(year) + '.txt'
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
