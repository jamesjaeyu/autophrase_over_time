
"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import time
import re
from glob import glob
import Levenshtein as Lv # Used to calculate string similarity
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


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


def process_seg(infolder):
    """
    Processes phrasal segmentation results to extract the group of phrases
    from each paper. Each year range will be outputted to separate files.
    Outputs csv with columns: Phrases, Year Range
    Each row represents a single paper in the DBLP v10 dataset
    (Takes around 2.5 minutes to run)

    NOTE: Additional processing may be required (in a separate function)
          to remove low-quality phrases

    NOTE: The repo doesn't contain the segmentation.txt files since they are so large,
          so this function may need to be skipped in the 'all' target for run.py
          and instead use the output YEAR_segmented.csv files

    >>> process_seg('../results/dblp-v10-grouped')
    """
    def extract_phrases(line):
        """
        Processes a single line from the segmentation results.
        Extracts all phrases marked with <phrase> and </phrases> and returns
        them in a string, with each phrase separated by commas.
        """
        line = line.lower()
        phrases = []
        while line.find('<phrase>') != -1:
            start_idx = line.find('<phrase>')
            end_idx = line.find('</phrase>')
            phrase = line[start_idx+8:end_idx]
            phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
            phrases.append(phrase)
            line = line[end_idx+9:]
        phrases = ','.join(phrases)
        return phrases

    start = time.time()
    # Obtains filepaths for all segmentation.txt files
    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/segmentation.txt')

    # Processes each segmentation.txt file
    for fp in filepaths:
        year = fp.split('/')[3]
        df = pd.read_csv(fp, sep='\n', header=None, names=['Phrases'])
        df['Year Range'] = [year] * len(df)
        df['Phrases'] = df.apply(lambda x: extract_phrases(x['Phrases']), axis=1)
        outpath = infolder + '/' + year + '_segmented.csv'
        df.to_csv(outpath)
    end = time.time()
    return end - start


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


def baseline_model(fp):
    """
    Baseline model using DecisionTreeClassifier

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

    Label (y):
    ---------------
    Year (int)
        THe year the phrase belongs to.

    Returns
    -------
    float
        The mean absolute difference between actual and predicted years on the test set.

    >>> baseline_model('../results/dblp-v10-phrases-uniquebyyear.csv')
    """
    # Reads in unique by year phrases
    phrases = pd.read_csv(fp, index_col=0)
    # Only keeps phrases from 1968+ and drops any null values
    phrases = phrases[phrases['Year'] >= 1968]
    phrases = phrases.dropna()

    # Creates num_years column: the number of years a phrase has shown up in
    counts = phrases.groupby('Phrase').size()
    phrases['num_years'] = phrases.apply(lambda x: counts[x['Phrase']], axis=1)

    # Creates pipeline
    std_pipe = Pipeline([('scale', StandardScaler())])
    ohe_pipe = Pipeline([('one-hot', OneHotEncoder(handle_unknown='ignore'))])
    ct = ColumnTransformer(transformers=[('ohe', ohe_pipe, ['Phrase']),
                                        ('scale', std_pipe, ['num_years']),
                                        ('keep', 'passthrough', ['Phrase Quality'])
                                        ])
    pl = Pipeline([('transform', ct), ('classifier', DecisionTreeClassifier())])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(phrases[['Phrase', 'num_years', 'Phrase Quality']],
                                                        phrases['Year'],
                                                        random_state=1)
    # Trains model
    pl.fit(X_train, y_train)

    return pl.score(X_test, y_test)
    # Runs predictions on test set
    #X_test['Predicted Year'] = pl.predict(X_test)

    # Creates Abs Year Diff column: the absolute difference between actual and predicted year
    #X_test['Year'] = y_test
    #X_test['Abs Year Diff'] = abs(X_test['Year'] - X_test['Predicted Year'])

    # Returns the average absolute difference - want this metric to be close to 0
    #return X_test['Abs Year Diff'].mean()


def refined_model():
    return
