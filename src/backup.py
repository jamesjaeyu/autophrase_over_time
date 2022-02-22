"""
DSC180B
Q2 Project
Backup for any unused functions
"""
import pandas as pd
import time
import re
from glob import glob

# NOTE: model_generation.py functions for processing phrasal segmentation results
# Using pandas is far more efficient than reading through the .txt files line by line
def process_seg(infolder):
    """
    Processes segmentation.txt files in all subfolders of infolder

    Creates csv with columns: Phrases, Year, Frequency
    Each entry of Phrases will contain all phrases within a single paper

    TODO: Issue with low-quality phrases being included in output. This can be
          resolved during processing or afterwards.

    >>> process_seg('../results/dblp-v10-grouped')
    """
    start = time.time()
    # Obtains filepaths for all segmentation.txt files
    filepaths = []
    for i in range(1950, 2019, 5):
        if i == 1950 or i == 1955:
            subfolder = '1950-1959'
        elif i == 2015:
            subfolder = '2015-2017'
        else:
            subfolder = str(i) + '-' + str(i+4)
        filepaths.append(infolder + '/' + subfolder + '/segmentation.txt')

    df = pd.DataFrame(columns=['Phrases', 'Year'])
    for fp in filepaths:
        year = fp.split('/')[3]
        file = open(fp)
        # Each line in the file represents a single paper's title + abstract
        for line in file:
            phrases = []
            line = line.lower()
            # Adds marked phrases to the data list until we have no more phrases
            while line.find('<phrase>') != -1:
                start_idx = line.find('<phrase>')
                end_idx = line.find('</phrase>')
                phrase = line[start_idx+8:end_idx]
                phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
                phrases.append(phrase)
                line = line[end_idx+9:]
            # Phrases are separated by commas
            phrases = ','.join(phrases)
            df.loc[len(df.index)] = [phrases, year]
    outpath = infolder + '/dblp-v10-grouped-seg.csv'
    df.to_csv(outpath)
    end = time.time()
    return end - start

def process_seg_alt(infolder):
    """
    Alternate approach to processing segmentation.txt files
    Outputs separate csvs for each year range, rather than all in a single csv
    (Takes around 7.5 hours to run)

    >>> process_seg_alt('../results/dblp-v10-grouped')
    """
    start = time.time()
    # Obtains filepaths for all segmentation.txt files
    filepaths = []
    for i in range(1950, 2019, 5):
        if i == 1950 or i == 1955:
            subfolder = '1950-1959'
        elif i == 2015:
            subfolder = '2015-2017'
        else:
            subfolder = str(i) + '-' + str(i+4)
        filepaths.append(infolder + '/' + subfolder + '/segmentation.txt')

    for fp in filepaths:
        df = pd.DataFrame(columns=['Phrases', 'Year'])
        file = open(fp)
        year = fp.split('/')[3]
        # Each line in the file represents a single paper's title + abstract
        for line in file:
            phrases = []
            line = line.lower()
            # Adds marked phrases to the data list until we have no more phrases
            while line.find('<phrase>') != -1:
                start_idx = line.find('<phrase>')
                end_idx = line.find('</phrase>')
                phrase = line[start_idx+8:end_idx]
                phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
                phrases.append(phrase)
                line = line[end_idx+9:]
            # Phrases are separated by commas
            phrases = ','.join(phrases)
            df.loc[len(df.index)] = [phrases, year]
        outpath = infolder + '/segmentation/' + year + '_segmented.csv'
        df.to_csv(outpath)
    end = time.time()
    return end - start


# NOTE: model_generation.py functions for processing AutoPhrase results
def obtain_phrases(infolder, unique_by_year=False):
    """
    Given the folder path containing AutoPhrase results by year, read in
    the quality phrases (multi >= 0.6, single >= 0.8) and output a csv.
    Columns: Phrase Quality, Phrase, Year

    unique_by_year determines if we look at unique phrases overall, or by year.
    - False means phrases across all years must be unique. So the earliest instance
        will be the only time the phrase shows up. (takes around 1 minute to run)
    - True means phrases just have to be unique per year. So there can be duplicates
        across multiple years. (takes around 20 minutes to run)

    TODO: Can try messing around with score threshold values

    >>> obtain_phrases('../results/dblp-v10', False)
    >>> obtain_phrases('../results/dblp-v10', True)
    """
    start = time.time()
    df = pd.DataFrame(columns=['Phrase Quality', 'Phrase', 'Year'])
    if not unique_by_year: # Set is maintained across all years
        phrases = set()
    for year in range(1960, 2018):
        filepath = infolder + '/' + str(year) + '/AutoPhrase.txt'
        if unique_by_year: # Set is reset every year
            phrases = set()
        try:
            file = open(filepath, 'r')
            for line in file:
                line = line.strip().split('\t')
                phrase = line[1]
                if phrase in phrases:
                    continue
                num_words = len(line[1].split())
                score = float(line[0])
                if (num_words > 1 and score >= 0.6) or (num_words == 1 and score >= 0.8):
                    df.loc[len(df.index)] = [score, phrase, year]
                    phrases.add(phrase)
            file.close()
        except:
            continue
    if unique_by_year:
        df.to_csv('../results/dblp-v10-phrases-uniquebyyear.csv')
    else:
        df.to_csv('../results/dblp-v10-phrases-unique.csv')
    end = time.time()
    return end - start


def obtain_phrases_alt(infolder, threshold):
    """
    Processes the grouped AutoPhrase results and outputs individual phrases to a csv
    (Takes around 18 minutes to run)

    infolder: The folder containing the dblp-v10-grouped AutoPhrase results
    threshold: Tuple containing the single & multi word quality minimums

    >>> obtain_phrases('../results/dblp-v10', (0.8, 0.5))
    >>> obtain_phrases('../results/dblp-v10-grouped', (0.8, 0.5))
    """
    start = time.time()
    df = pd.DataFrame(columns=['Phrase Quality', 'Phrase', 'Year'])

    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/AutoPhrase.txt')

    for fp in filepaths:
        try:
            file = open(fp, 'r')
            year = fp.split('/')[3]
            for line in file:
                line = line.strip().split('\t')
                phrase = line[1]
                num_words = len(line[1].split())
                score = float(line[0])
                if (num_words == 1 and score >= threshold[0]) or (num_words > 1 and score >= threshold[1]):
                    df.loc[len(df.index)] = [score, phrase, year]
            file.close()
        except:
            continue
    outpath = infolder + '/phrases.csv'
    #'../results/dblp-v10-grouped/dblp-v10-grouped-phrases.csv'
    df.to_csv(outpath)
    end = time.time()
    return end - start


# NOTE: From sort_jsons.py
import json

#Clear existing files before sorting json files
for x in range(1935,2020):
	filename = str(x) + '.txt'
	open(filename, 'w').close()

#For use with the V10 DBLP database from https://www.aminer.org/citation
#This data set is split into four .json files, which are imported in sequence
#Note: this takes a long time to run, so it's possible to comment out multiple sections of the data set for quicker running and testing
file = open('dblp-ref-0.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-1.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-2.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

file = open('dblp-ref-3.json')
for l in file:
	data = json.loads(l)
	year = data['year']
	yeartxt = str(year) + ".txt"
	writefile = open(yeartxt, "a", encoding="utf-8")
	if 'abstract' in data.keys():
		writefile.writelines(data['abstract'])
		writefile.writelines("\n")
	if 'title' in data.keys():
		writefile.writelines(data['title'])
		writefile.writelines("\n")
file.close()

