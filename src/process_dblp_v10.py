"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import json
import time
import re

def print_stats(time, papers, issues, empty, year):
    print(str(time) + ' seconds has elapsed since start of function')
    print(str(papers) + ' papers processed')
    print(str(issues) + ' number of papers with json formatting issues (' + str(issues/papers)[:5] + ')')
    print(str(empty) + ' number of papers with empty abstracts (' + str(empty/papers)[:5] + ')')
    print(str(year) + ' number of papers with invalid/irrelevant years < 1950, > 2022')


def process_v10_txt(infolder):
    """
    Processes DBLP v10 dataset
    Outputs aggregate .txt files by year (only titles + abstracts)

    Takes around 8-9 minutes to process all papers

    #>>> process_v10_txt('../data/dblp-v10')
    """
    start = time.time()
    filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outfolder = infolder + '/txt/'

    num_no_abstract, num_no_title, num_papers, num_issues, num_year = 0, 0, 0, 0, 0

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
                    # if len(data['title']) == 0:
                    #     num_no_title += 1
                    #     continue
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    outfile.write(title + '\n')
                else:
                    num_no_title += 1
                    continue
            except:
                num_issues += 1
            # For testing purposes - printing info + stopping early
            # if num_papers >= 100:
            #     end = time.time()
            #     time_elapsed = end - start
            #     print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
            #     return
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
    return


def process_v10_csv(infolder):
    """
    Outputs aggregate .csv files by year (titles + abstracts)

    >>> process_v10_csv('../data/dblp-v10')
    """
    start = time.time()
    filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outfolder = infolder + '/csv/'
    years = set()

    num_no_abstract, num_no_title, num_papers, num_issues, num_year = 0, 0, 0, 0, 0

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
            # For testing purposes - printing info + stopping early
            if num_papers >= 100:
                end = time.time()
                time_elapsed = end - start
                print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
                return
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
    return


def process_v10_txt_agg(infolder):
    """
    #>>> process_v10_txt_agg('../data/dblp-v10')
    """
    start = time.time()
    filepaths = [infolder + '/dblp-ref-' + str(num) + '.json' for num in range(4)]
    outpath = infolder + '/agg/content_FULL.txt'

    num_no_abstract, num_no_title, num_papers, num_issues, num_year = 0, 0, 0, 0, 0

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
                    # if len(data['title']) == 0:
                    #     num_no_title += 1
                    #     continue
                    title = data['title']
                    title = re.sub(r'[^A-Za-z0-9- ]+', '', title)
                    outfile.write(title + '\n')
                else:
                    num_no_title += 1
                    continue
            except:
                num_issues += 1
            # For testing purposes - printing info + stopping early
            # if num_papers >= 100:
            #     end = time.time()
            #     time_elapsed = end - start
            #     print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
            #     return
        file.close()
    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
    return
