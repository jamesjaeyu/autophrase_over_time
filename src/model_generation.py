"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import time
import Levenshtein as Lv # Used to calculate string similarity
from sklearn.model_selection import train_test_split

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


def create_model():
    """
    Idea 1: If we have multiple word2vec models for each year/era, we can
        look at the similarity of the phrases of an individual paper against each model.
    But this requires obtaining the phrasal segmentation results

    Idea 2: Using correlation between phrases of an individual paper and each year/era's phrases
        Classify based on the highest average correlation

    """
    return
