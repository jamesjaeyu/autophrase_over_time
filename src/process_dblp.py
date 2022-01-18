"""
DSC180B
Q2 Project
"""
import pandas as pd
from random import sample
import sys
import json
import time

def process_json(inpath):
    """
    Process the inpath of DBLP v13 dataset
    Writes out to aggregate files by year

    Only care about: title, year, keywords, abstract
    When we encounter one of these tags, we want to keep processing lines until we reach an invalid tag

    When running on the entire dataset
    On local machine (James) it took 1885 seconds (31.4 minutes)
    1,445,201 out of 5,329,485 papers had issues with json.loads() formatting
        27.1% of papers had issues
    TODO: look at the content of papers that had issues and try to fix general cases

    #>>> process_json('../data/dblpv13.json')
    """
    start = time.time()
    data = []
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    invalid_tags = ['"venue" :', '"_id" :', '"type" :', '"raw" :', '"raw_zh" :',
                    '"n_citation" :', '"page_start" :', '"page_end" :',
                    '"lang" :', '"volume" :', '"issue" :', '"issn" :',
                    '"isbn" :', '"doi" :', '"pdf" :', '"url" :', '"references" :']
    valid_tag = True
    num_papers = 0
    years = set()
    skip_paper = False
    num_issues = 0

    content = '{'
    # Keeps track of the start of processing a block (individual paper)
    start_ind = False

    with open(inpath, 'r', encoding='utf-8') as f:
        for line in f:
            # Indicates start of block
            if line == '{ \n':
                start_ind = True

            # Indicates end of block
            if line == '},\n':
                # Doesn't process the first paper (only contains info about structure)
                if first_line:
                    first_line = False
                    content = '{'
                    start_ind = False
                    continue
                # If there is an indicator to skip the paper
                if skip_paper:
                    content = '{'
                    skip_paper = False
                    start_ind = False
                    continue

                # Cleans up formatting of block
                content += '}\n'
                content = content.replace(' :', ':').replace(',', ', ')
                # Prevents issue with additional comma at end of dict
                if content[-4:-2] == ', ':
                    content = content[:-4] + '}\n'
                # Fix for issue with double beginning brackets
                if content[:2] == '{{':
                    content = content[1:]

                num_papers += 1
                try:
                    formatted = json.loads(content)
                    # Writing out titles + abstracts to aggregated year .txt file
                    fp1 = '../data/dblp-v13/content_' + str(formatted['year']) + '.txt'
                    f1 = open(fp1, 'a')
                    f1.write(formatted['title'] + ' ' + formatted['abstract'] + ' ')
                    f1.close()
                    # Writing out DBLP keywords to aggregated year .txt file
                    fp2 = '../data/dblp-v13/keywords_' + str(formatted['year']) + '.txt'
                    f2 = open(fp2, 'a')
                    keywords = [word.replace(' ', '_') for word in formatted['keywords']]
                    f2.write(' '.join(keywords) + ' ')
                    f2.close()
                except:
                    # Issue can occur with json.loads() function due to formatting
                    num_issues += 1

                # For testing purposes - printing info + stopping early
                # if num_papers == 100000:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print(str(time_elapsed) + ' seconds has elapsed since start of function')
                #     print(str(num_papers) + ' papers processed')
                #     print(str(num_issues) + ' number of papers with issues')
                #     return

                # Reset parameters
                start_ind = False
                content = '{'

            # Processes individual lines once we have found the start indicator
            if start_ind:
                line = line.strip()

                # We only care about title, abstract, keywords, year, fos tags
                if any([tag in line for tag in tags]):
                    valid_tag = True
                elif ' : ' in line:
                    # Catches invalid tags
                    valid_tag = False
                elif '.fr' in line:
                    # Had issues with french papers being included
                    skip_paper = True

                if not valid_tag:
                    continue

                if 'NumberInt' in line:
                    line = line.replace('NumberInt(', '')
                    line = line.strip()[:-2] + ','
                if line[-3:] == ': ,':
                    line = line.replace(': ,', ': null')
                if line == '"type" : 1,':
                    continue
                content += line

    end = time.time()
    time_elapsed = end - start
    print(str(time_elapsed) + ' seconds has elapsed since start of function')
    print(str(num_papers) + ' papers processed')
    print(str(num_issues) + ' number of papers with issues')
    return


# TODO
# 1. Obtain the number of papers per year (value counts)
# 2. Output the title + abstracts to aggregated .txt files by year
# 3. Output unique keywords to aggregated .txt files by year
#    (Could also try including duplicates so we can see the most common keywords by year)
# 4. Implement year cutoff
# 5. Implement title/abstract length minimum thresholds


def process_json_csv(inpath):
    """
    Process the inpath of DBLP v13 dataset
    Writes out aggregate files (but as csvs)

    Only care about: title, year, keywords, abstract
    When we encounter one of these tags, we want to keep processing lines until we reach an invalid tag

    When running on the entire dataset
    On local machine (James) it took 1885 seconds (31.4 minutes)
    1,445,201 out of 5,329,485 papers had issues with json.loads() formatting
        27.1% of papers had issues
    TODO: look at the content of papers that had issues and try to fix general cases

    >>> process_json_csv('../data/dblpv13.json')
    """
    start = time.time()
    data = []
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    invalid_tags = ['"venue" :', '"_id" :', '"type" :', '"raw" :', '"raw_zh" :',
                    '"n_citation" :', '"page_start" :', '"page_end" :',
                    '"lang" :', '"volume" :', '"issue" :', '"issn" :',
                    '"isbn" :', '"doi" :', '"pdf" :', '"url" :', '"references" :']
    valid_tag = True
    num_papers = 0
    years = set()
    skip_paper = False
    num_issues = 0
    num_abs = 0
    num_year = 0

    content = '{'
    # Keeps track of the start of processing a block (individual paper)
    start_ind = False

    with open(inpath, 'r', encoding='utf-8') as f:
        for line in f:
            # Indicates start of block
            if line == '{ \n':
                start_ind = True

            # Indicates end of block
            if line == '},\n':
                # Doesn't process the first paper (only contains info about structure)
                if first_line:
                    first_line = False
                    content = '{'
                    start_ind = False
                    continue
                # If there is an indicator to skip the paper
                if skip_paper:
                    content = '{'
                    skip_paper = False
                    start_ind = False
                    continue

                # Cleans up formatting of block
                content += '}\n'
                content = content.replace(' :', ':').replace(',', ', ')
                # Prevents issue with additional comma at end of dict
                if content[-4:-2] == ', ':
                    content = content[:-4] + '}\n'
                # Fix for issue with double beginning brackets
                if content[:2] == '{{':
                    content = content[1:]

                num_papers += 1
                try:
                    formatted = json.loads(content)
                    keywords = ' '.join([word.replace(' ', '_') for word in formatted['keywords']])
                    title = formatted['title'].replace(',', '').replace('\n', '').replace('\\', '')
                    abstract = formatted['abstract'].replace(',', '').replace('\n','').replace('\\', '')
                    if len(abstract) == 0:
                        num_abs += 1
                        continue
                    if formatted['year'] < 1950 or formatted['year'] > 2022:
                        num_year += 1
                        continue
                    # Writing out data to csv
                    fp1 = '../data/dblp-v13/csv/' + str(formatted['year']) + '.csv'
                    f1 = open(fp1, 'a')
                    if formatted['year'] not in years:
                        f1.write('Title,Abstract,Keywords\n')
                    f1.write(title + ',' + abstract + ',' + keywords + '\n')
                    f1.close()
                    years.add(formatted['year'])
                except:
                    # Issue can occur with json.loads() function due to formatting
                    num_issues += 1

                # For testing purposes - printing info + stopping early
                # if num_papers == 1000:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print(str(time_elapsed) + ' seconds has elapsed since start of function')
                #     print(str(num_papers) + ' papers processed')
                #     print(str(num_issues) + ' number of papers with json formatting issues')
                #     print(str(num_abs) + ' number of papers with empty abstracts')
                #     print(str(num_year) + ' number of papers with invalid/irrelevant years < 1950, > 2022')
                #     return

                # Reset parameters
                start_ind = False
                content = '{'

            # Processes individual lines once we have found the start indicator
            if start_ind:
                line = line.strip()

                # We only care about title, abstract, keywords, year, fos tags
                if any([tag in line for tag in tags]):
                    valid_tag = True
                elif ' : ' in line:
                    # Catches invalid tags
                    valid_tag = False
                elif '.fr' in line:
                    # Had issues with french papers being included
                    skip_paper = True

                if not valid_tag:
                    continue

                if 'NumberInt' in line:
                    line = line.replace('NumberInt(', '')
                    line = line.strip()[:-2] + ','
                if line[-3:] == ': ,':
                    line = line.replace(': ,', ': null')
                if line == '"type" : 1,':
                    continue
                content += line

    end = time.time()
    time_elapsed = end - start
    print(str(time_elapsed) + ' seconds has elapsed since start of function')
    print(str(num_papers) + ' papers processed')
    print(str(num_issues) + ' number of papers with issues')
    print(str(num_abs) + ' number of papers with empty abstracts')
    print(str(num_year) + ' number of papers with invalid/irrelevant years < 1950, > 2022')
    return