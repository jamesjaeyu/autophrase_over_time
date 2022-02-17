"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import time
import re
import Levenshtein as Lv # Used to calculate string similarity
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


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


def obtain_phrases_grouped(infolder, threshold):
    """
    Processes the grouped AutoPhrase results and outputs individual phrases to a csv
    (Takes around 18 minutes to run)

    TODO: This function could be combined with obtain_phrases by using os to
    automatically find all subfolders within the infolder

    infolder: The folder containing the dblp-v10-grouped AutoPhrase results
    threshold: Tuple containing the single & multi word quality minimums

    >>> obtain_phrases_grouped('../results/dblp-v10-grouped', (0.8, 0.5))
    """
    start = time.time()
    df = pd.DataFrame(columns=['Phrase Quality', 'Phrase', 'Year'])

    filepaths = []
    for i in range(1950, 2019, 5):
        if i == 1950 or i == 1955:
            subfolder = '1950-1959'
        elif i == 2015:
            subfolder = '2015-2017'
        else:
            subfolder = str(i) + '-' + str(i+4)
        filepaths.append(infolder + '/' + subfolder + '/AutoPhrase.txt')

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
    df.to_csv('../results/dblp-v10-grouped/dblp-v10-grouped-phrases.csv')
    end = time.time()
    return end - start


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
    (Takes around 7.5 hours to fully run)

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
    # Runs predictions on test set
    X_test['Predicted Year'] = pl.predict(X_test)

    # Creates Abs Year Diff column: the absolute difference between actual and predicted year
    X_test['Year'] = y_test
    X_test['Abs Year Diff'] = abs(X_test['Year'] - X_test['Predicted Year'])

    # Returns the average absolute difference - want this metric to be close to 0
    return X_test['Abs Year Diff'].mean()


def refined_model():
    return