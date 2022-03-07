"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import numpy as np
import time
import re
import os
from glob import glob
import Levenshtein as Lv # Used to calculate string similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier


def obtain_phrases(infolder, threshold=(0.8,0.5)):
    """
    Outputs a csv containing consolidated AutoPhrase results.
    Columns: Phrase Quality, Phrase, Year, Num Words
    (Takes 8 seconds to run)

    infolder: Folder path containing AutoPhrase results by year
    threshold: Tuple containing floats for (single-word, multi-word) thresholds
               for phrase quality

    >>> obtain_phrases('../results/dblp-v10', (0.8, 0.5))
    >>> obtain_phrases('../results/dblp-v10-grouped', (0.8, 0.5))
    """
    start = time.time()
    # Gathers filepaths of each AutoPhrase.txt file
    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/AutoPhrase.txt')

    # Output dataframe
    out_df = pd.DataFrame(columns=['Phrase Quality', 'Phrase', 'Year', 'Num Words'])
    # Processes each AutoPhrase.txt file
    for fp in filepaths:
        year = fp.split('/')[-2]
        df = pd.read_csv(fp, sep='\t', header=None, names=['Phrase Quality', 'Phrase'])
        df = df.dropna()
        df['Year'] = [year] * len(df)
        # Number of words in the phrase
        df['Num Words'] = df['Phrase'].map(str.split).map(len)
        # Filters out single and multi-word phrases based on phrase quality
        filter_qual = lambda x: True if \
            (x['Num Words'] > 1 and x['Phrase Quality'] >= threshold[1]) \
            or (x['Num Words'] == 1 and x['Phrase Quality'] >= threshold[0]) \
            else False
        # Only keeps phrases above given threshold
        valid_idx = df.apply(filter_qual, axis=1)
        df = df[valid_idx]
        # Appends rows to output dataframe
        out_df = out_df.append(df, ignore_index=True)
    # Outputs to infolder as phrases.csv
    outpath = infolder + '/phrases.csv'
    out_df.to_csv(outpath)
    end = time.time()
    return end - start


def process_seg(infolder, outfolder, phrases_fp=None, method='general'):
    """
    Processes phrasal segmentation .txt results
    Outputs .csv files with columns: Phrases, Year Range
    Each row represents a single paper

    method='general': All phrases from phrasal segmentation results are included in output
        phrases_fp can be a random input - it will not be used
    method='model': Only high-quality phrases, no duplicates in each row
    method='gephi': Only multi-word, high-quality phrases, no duplicates in each row

    >>> process_seg('../results/dblp-v10-grouped', '../results/dblp-v10-grouped')

    >>> process_seg('../results/dblp-v10-grouped', \
                    '../results/dblp-v10-model', \
                    '../results/dblp-v10-grouped/phrases.csv', \
                    'model')

    >>> process_seg('../results/dblp-v10-grouped', \
                    '../results/dblp-v10-gephi', \
                    '../results/dblp-v10-grouped/phrases.csv', \
                    'gephi')
    """
    start = time.time()
    if method not in ['general', 'model', 'gephi']:
        print('Choose a valid process_seg method: general, model, gephi')
        return

    # Only needed if the method is 'model' or 'gephi'
    if method != 'general':
        # Reading in AutoPhrase results csv
        df_phrases = pd.read_csv(phrases_fp, index_col=0)
        # Set containing all the high-quality phrases from AutoPhrase results
        # (single-words with quality >= 0.8, multi >= 0.5)
        unique_phrases = set(df_phrases['Phrase'].values)

    # Getting filepaths for segmentation.txt files
    infolder = '../results/dblp-v10-grouped'
    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/segmentation.txt')

    def extract_phrases(line):
        """
        Helper function for processing phrasal segmentation .txt results
        Outputs a string containing phrases in line, separated by commas
        """
        line = line.lower()
        if method == 'general':
            out = []
        else:
            out = set()
        # Processes the line until there are no phrases left
        while line.find('<phrase>') != -1:
            start_idx = line.find('<phrase>')
            end_idx = line.find('</phrase>')
            # Obtains text between phrase markers
            phrase = line[start_idx+8:end_idx]
            line = line[end_idx+9:]
            # Removes any non-alphanumeric characters
            phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
            phrase = re.sub(r'-', ' ', phrase)
            # Adds phrase to output
            if method == 'general':
                out.append(phrase)
            else:
                # Ensures we only keep multi-word phrases for 'gephi' method
                # if method == 'gephi':
                #     num_words = len(phrase.split())
                #     if num_words == 1:
                #         continue

                # If the phrase is contained within the AutoPhrase results
                # it is a high-quality phrase, so we can add it
                if phrase in unique_phrases:
                    out.add(phrase)
        # Output will be all the phrases in a single string, separated by commas
        if method != 'general':
            out = list(out)
        out = ','.join(out)
        return out

    # Processing the segmentation.txt files for each year range
    # Outputs into separate .csv files for each year range
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for fp in filepaths:
        year = fp.split('/')[3]
        df = pd.read_csv(fp, sep='\n', header=None, names=['Phrases'])
        df['Year Range'] = [year] * len(df)
        df['Phrases'] = df.apply(lambda x: extract_phrases(x['Phrases']), axis=1)
        outpath = outfolder + '/' + year + '_segmented_' + method + '.csv'
        df.to_csv(outpath)
    end = time.time()
    return str(end-start) + ' seconds to run'


def phrase_counts(infolder):
    """
    Using the segmentation results, returns a dictionary with the phrase counts
    across each year range.
    Dictionary format will be {Year Range: {phrase: count}}

    >>> phrase_counts('../results/dblp-v10-grouped')
    """
    def add_counts(x):
        """
        Helper function for processing segmented.csv dataframes
        Modifies the counts output dictionary
        """
        phrases = x['Phrases'].split(',')
        year = x['Year']
        for phrase in phrases:
            if phrase not in counts[year]:
                counts[year][phrase] = 0
            counts[year][phrase] += 1

    # Obtains filepaths for all segmented.csv files
    subfolders = glob(infolder + '/*.csv')
    filepaths = list(filter(lambda x: 'segmented' in x, subfolders))

    # Creates dataframe with all segmentation.csv data combined
    seg = pd.DataFrame(columns=['Phrases', 'Year'])
    for fp in filepaths:
        df = pd.read_csv(fp, index_col=0)
        df = df.dropna()
        #df['Num Phrases'] = df.apply(lambda x: len(x['Phrases'].split(',')), axis=1)
        #df = df.drop('Phrases', axis=1)
        seg = seg.append(df, ignore_index=True)

    # Output dictionary
    counts = {}
    for yr in seg['Year'].unique():
        counts[yr] = {}

    # Helper function will modify the counts dictionary
    seg.apply(add_counts, axis=1)

    # Sorts the inner dictionaries of counts in descending order based on frequency
    for key, val in counts.items():
        counts[key] = dict(sorted(val.items(), key=lambda item: item[1], reverse=True))

    # Dictionary that only contains multi-word phrases and counts
    multi_counts = {}
    for year_range, phrase_counts in counts.items():
        multi_counts[year_range] = {key: val for key, val in phrase_counts.items() if len(key.split()) > 1}

    # Creates counts dictionary but with percent as values, rather than raw frequency
    counts_per = {}
    for year_range, phrase_counts in counts.items():
        total_count_yr = sum(phrase_counts.values())
        prop_counts = {}
        for key, val in phrase_counts.items():
            prop_counts[key] = (val / total_count_yr) * 100
        counts_per[year_range] = prop_counts


def find_similar(input_phrase, fp):
    """
    Given an input_phrase, return the most similar phrase from the AutoPhrase
    results, along with all of the years in which that phrase has appeared.

    Output format: (distance, most similar phrase, [years where the MSP appears])

    >>> find_similar('convolutional neural networks', '../results/dblp-v10-phrases-uniquebyyear.csv')
    (0.0, 'convolutional neural networks', [2012, 2013, 2014, 2015, 2016, 2017])
    """
    df = pd.read_csv(fp)
    df['Dist'] = df.apply(lambda x: Lv.distance(input_phrase, x['Phrase'])
                        if isinstance(x['Phrase'], str) else float('inf'), axis=1)
    df_sorted = df.sort_values('Dist')
    closest = df_sorted['Phrase'][0]
    closest_dist = df_sorted['Dist'][0]
    years = list(df[df['Phrase'] == closest]['Year'])
    return (closest_dist, closest, years)


def generate_model(fp):
    """
    fp: Filepath containing AutoPhrase phrase mining results csv

    Features (x):
    -------------
    Phrase (str)
        uses OneHotEncoder. Potential issues can arise when a phrase that
        is not in the training set is passed into the model. Errors resolved by the
        'handle_unknown' parameter in the ohe_pipe
    num_years (int)
        uses StandardScaler. Tells us the number of years a phrase has appeared in
    Phrase Quality (float)
        no modification. May need to normalize in the future. Also, the
        'phrases' dataframe only contains high-quality phrases (multi >= 0.6, single >= 0.8)
    Num Words (int)
        Number of words in the phrase

    Label (y):
    ---------------
    Year (int)
        The year the phrase belongs to.

    Returns
    -------
    float
        Accuracy on the test set

    >>> refined_model('../results/dblp-v10-grouped/phrases.csv')
    """
    df = pd.read_csv(fp, index_col=0)
    df = df.dropna()
    # Creates 'num_years' column - number of year ranges a phrase has appeared in
    phr_counts = df.groupby('Phrase').size()
    df['num_years'] = df.apply(lambda x: phr_counts[x['Phrase']], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[['Phrase', 'num_years', 'Phrase Quality']],
                                                        df['Year'])

    # Creates pipeline
    std_pipe = Pipeline([('scale', StandardScaler())])
    ohe_pipe = Pipeline([('one-hot', OneHotEncoder(handle_unknown='ignore'))])
    ct = ColumnTransformer(transformers=[('ohe', ohe_pipe, ['Phrase']),
                                        ('scale', std_pipe, ['num_years']),
                                        ('keep', 'passthrough', ['Phrase Quality', 'Num Words'])
                                        ])
    pl = Pipeline([('transform', ct), ('classifier', AdaBoostClassifier(n_estimators=100,
                                                                        learning_rate=1.1))])

    # Trains model
    pl.fit(X_train, y_train)

    # Baseline accuracy - most popular label
    base_acc = (y_test == '2010-2014').mean()

    return pl.score(X_test, y_test)
