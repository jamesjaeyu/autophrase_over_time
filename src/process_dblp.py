"""
DSC180B
Q2 Project
"""
import pandas as pd
import json
import time

def print_stats(time, papers, issues, empty, year):
    print(str(time) + ' seconds has elapsed since start of function')
    print(str(papers) + ' papers processed')
    print(str(issues) + ' number of papers with json formatting issues (' + str(issues/papers) + '%)')
    print(str(empty) + ' number of papers with empty abstracts (' + str(empty/papers) + '%)')
    print(str(year) + ' number of papers with invalid/irrelevant years < 1950, > 2022 (' + str(year/papers) + '%)')


def process_v10_txt(infolder):
    """
    Processes DBLP v10 dataset
    Outputs aggregate .txt files by year (only titles + abstracts)

    >>> process_v10('../data/dblp-v10')
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
                    outfile.write(data['abstract'] + ' ')
                else:
                    num_no_abstract += 1
                    continue
                if 'title' in data.keys():
                    if len(data['title']) == 0:
                        num_no_title += 1
                        continue
                    outfile.write(data['title'] + ' ')
                else:
                    num_no_title += 1
                    continue
                outfile.close()
            except:
                num_issues += 1


    end = time.time()
    time_elapsed = end - start
    print_stats(time_elapsed, num_papers, num_issues, num_no_abstract, num_year)
    return


def process_v10_csv(infolder):
    """
    Outputs aggregate .csv files by year (titles + abstracts)

    >>> process_v10_csv('../data/dblp-v10')
    """
    return


def process_v10_txt_agg(infolder):
    """
    >>> process_v10_txt_agg('../data/dblp-v10')
    """
    return


def process_json_txt(inpath):
    """
    Writes out content_YEAR.txt and keywords_YEAR.txt files
    content contains titles + abstracts for that year
    keywords contains all of the keywords for that year (duplicates allowed)

    TODO: Fix issues with json.loads() errors
    TODO: Look into more filters - based on year, title length, number of keywords

    #>>> process_json_txt('../data/dblpv13.json')
    """
    start = time.time()
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    valid_tag = True
    skip_paper = False
    num_papers, num_issues, num_abs, num_year = 0, 0, 0, 0

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
                    year = formatted['year']
                    if len(abstract) == 0:
                        num_abs += 1
                        continue
                    if year < 1950 or year > 2022:
                        num_year += 1
                        continue

                    # Writing out titles + abstracts to aggregated year .txt file
                    fp1 = '../data/dblp-v13/content_' + str(year) + '.txt'
                    f1 = open(fp1, 'a')
                    f1.write(title + ' ' + abstract + ' ')
                    f1.close()
                    # Writing out DBLP keywords to aggregated year .txt file
                    fp2 = '../data/dblp-v13/keywords_' + str(year) + '.txt'
                    f2 = open(fp2, 'a')
                    f2.write(keywords + ' ')
                    f2.close()
                except:
                    # Issue can occur with json.loads() function due to formatting
                    num_issues += 1

                # For testing purposes - printing info + stopping early
                # if num_papers == 100000:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
                #     return

                # Reset parameters
                start_ind = False
                content = '{'

            # Processes individual lines once we have found the start indicator
            if start_ind:
                # TODO: Handle issues in line processing that cause issues with json.loads

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
    print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
    return


def process_json_csv(inpath):
    """
    Writes out YEAR.csv files
    Columns are Title, Abstract, Keywords
    Written to the dblp-v13/csv folder

    TODO: Fix issue where pandas.read_csv has issues
        (try creating dataframes first, then export to csv)

    >>> process_json_csv('../data/dblpv13.json')
    """
    start = time.time()
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    valid_tag = True
    num_papers, num_issues, num_abs, num_year = 0, 0, 0, 0
    years = set()
    skip_paper = False

    content = '{'
    # Keeps track of the start of processing a block (individual paper)
    start_ind = False

    # Dictionary contains year:dataframe key:val pairs
    out = {}

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
                    year = formatted['year']
                    if len(abstract) == 0:
                        num_abs += 1
                        continue
                    if year < 1950 or year > 2022:
                        num_year += 1
                        continue

                    # TODO: (new) Adding information to dictionary dataframe
                    if year not in out:
                        out[year] = pd.DataFrame(columns=['Title', 'Abstract', 'Keywords'])
                    out[year].loc[len(out[year].index)] = [title, abstract, keywords]

                    # Writing out data to csv
                    # fp1 = '../data/dblp-v13/csv/' + str(year) + '.csv'
                    # f1 = open(fp1, 'a')
                    # if year not in years:
                    #     f1.write('Title,Abstract,Keywords\n')
                    # f1.write(title + ',' + abstract + ',' + keywords + '\n')
                    # f1.close()
                    # years.add(year)
                except:
                    # Issue can occur with json.loads() function due to formatting
                    num_issues += 1

                # For testing purposes - printing info + stopping early
                # if num_papers == 1000:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
                #     for key, val in out.items():
                #         outpath = '../data/dblp-v13/testcsv/' + str(key) + '.csv'
                #         val.to_csv(outpath)
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
    print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
    # Writes out DataFrames in dictionary to outpath folder
    for key, val in out:
        outpath = '../data/dblp-v13/testcsv/' + str(key) + '.csv'
        val.to_csv(outpath)
    return


def process_json_txt_agg(inpath):
    """
    Writes out aggregated content.txt and keywords.txt - all combined together
    Written to the dblp-v13/agg folder

    #>>> process_json_txt_agg('../data/dblpv13.json')
    """
    start = time.time()
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    valid_tag = True
    skip_paper = False
    num_papers, num_issues, num_abs, num_year = 0, 0, 0, 0

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
                    year = formatted['year']
                    if len(abstract) == 0:
                        num_abs += 1
                        continue
                    if year < 1950 or year > 2022:
                        num_year += 1
                        continue

                    # Writing out titles + abstracts to aggregated year .txt file
                    fp1 = '../data/dblp-v13/agg/content.txt'
                    f1 = open(fp1, 'a')
                    f1.write(title + ' ' + abstract + ' ')
                    f1.close()
                    # Writing out DBLP keywords to aggregated year .txt file
                    fp2 = '../data/dblp-v13/agg/keywords.txt'
                    f2 = open(fp2, 'a')
                    f2.write(keywords + ' ')
                    f2.close()
                except:
                    # Issue can occur with json.loads() function due to formatting
                    num_issues += 1

                # For testing purposes - printing info + stopping early
                # if num_papers == 100000:
                #     end = time.time()
                #     time_elapsed = end - start
                #     print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
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
    print_stats(time_elapsed, num_papers, num_issues, num_abs, num_year)
    return
