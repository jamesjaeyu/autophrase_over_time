"""
DSC180B
Q2 Project
Processing arxivData
"""
import pandas as pd
import json
import time
import re
import os

def print_stats(time, papers):
    print(str(time) + ' seconds has elapsed since start of function')
    print(str(papers) + ' papers processed')


def process_arxiv_txt(infolder):
    """
    Processes Arxiv dataset
    Outputs aggregate .txt files by year (only titles + abstracts)
    Takes around 30 seconds to process all papers
    >>> process_arxiv_txt('../data/arxiv')
    """
    start = time.time()
    fp = infolder + '/arxivData.json'
    outfolder = infolder + '/txt/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    num_papers = 0

    file = pd.read_json(fp)
    for index, data in file.iterrows():
        num_papers += 1
        year = data['year']
        outpath = outfolder + str(year) + '.txt'
        outfile = open(outpath, 'a')
        abstract = data['summary']
        abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
        outfile.write(abstract + '\n')
        title = data['title']
        title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
        outfile.write(title + '\n')
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers)
    return


def process_arxiv_csv(infolder):
    """
    Outputs aggregate .csv files by year (only titles + abstracts)
    >>> process_arxiv_csv('../data/arxiv')
    """
    start = time.time()
    fp = infolder + '/arxivData.json'
    outfolder = infolder + '/csv/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    num_papers = 0
    years = set()

    file = pd.read_json(fp)
    for index, data in file.iterrows():
        num_papers += 1
        year = data['year']
        outpath = outfolder + str(year) + '.csv'
        outfile = open(outpath, 'a')
        if year not in years:
            outfile.write('Title,Abstract\n')
        abstract = data['summary']
        abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
        title = data['title']
        title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
        outfile.write(title + ',' + abstract + '\n')
        years.add(year)
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers)
    return


def process_arxiv_txt_agg(infolder):
    """
    Processes Arxiv dataset into a single txt file (only titles + abstracts)
    >>> process_arxiv_txt_agg('../data/arxiv')
    """
    start = time.time()
    fp = infolder + '/arxivData.json'
    outfolder = infolder + '/agg/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    num_papers = 0

    file = pd.read_json(fp)
    for index, data in file.iterrows():
        num_papers += 1
        year = data['year']
        outpath = outfolder + 'content_FULL.txt'
        outfile = open(outpath, 'a')
        abstract = data['summary']
        abstract = re.sub(r'[^A-Za-z0-9- ]+', '', abstract)
        outfile.write(abstract + '\n')
        title = data['title']
        title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
        outfile.write(title + '\n')
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers)
    return
