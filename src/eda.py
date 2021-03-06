"""
DSC180B
Q2 Project
EDA for DBLP v10 and arXiv datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
from glob import glob

def generate_figures(outfolder):
    """
    Main function for outputting all figures

    >>> generate_figures('../results/figures')
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    v10_papers_per_year('data/dblp-v10/dblp-v10-counts.csv', outfolder)
    autophrase_figures('results/dblp-v10-grouped/phrases.csv', outfolder)
    return


def v10_papers_per_year(fp, outfolder):
    """
    Takes in filepath for the DBLP v10 counts per year .csv file,
    as well as outfolder for graph generated
    """
    v10_counts = pd.read_csv(fp, index_col=0)
    plt.bar(v10_counts['Year'], v10_counts['Count'])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Number of papers per year (DBLP v10)')
    fn = '/v10_papers_per_year.png'
    plt.savefig(outfolder + fn)
    return


def autophrase_figures(fp, outfolder):
    """
    Generates various figures from the AutoPhrase and phrasal segmentation results.
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    # Reads in AutoPhrase results csv
    fp = 'results/dblp-v10-grouped/phrases.csv'
    df = pd.read_csv(fp, index_col=0)

    # Bar chart for average number of words in phrases
    avg_num_words = df.groupby('Year')['Num Words'].mean()
    avg_num_words.plot(kind='barh',
                       xlabel='Year Range',
                       title='Average number of words per phrase in each year range')
    plt.xlabel('Average number of words in each phrase')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.2)
    fn = '/bar_avg_phrase_length.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Histogram of phrase length
    df['Num Words'].plot.hist(title='Distribution of number of words in phrases')
    plt.xlabel('Number of words in phrase')
    plt.subplots_adjust(left=0.2)
    fn = '/hist_phrase_length.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Reads in phrasal segmentation csvs
    infolder = 'results/dblp-v10-grouped'
    subfolders = glob(infolder + '/*.csv')
    subfolders = list(filter(lambda x: 'segmented' in x, subfolders))
    out = pd.DataFrame(columns=['Year', 'Num Phrases'])
    for fp in subfolders:
        df = pd.read_csv(fp, index_col=0)
        df = df.dropna()
        df['Num Phrases'] = df.apply(lambda x: len(x['Phrases'].split(',')), axis=1)
        df = df.drop('Phrases', axis=1)
        out = out.append(df, ignore_index=True)

    # Bar chart for average number of phrases identified by AutoPhrase
    avg_phrases = out.groupby('Year')['Num Phrases'].mean()
    avg_phrases.plot(kind='barh',
                     xlabel='Year Range',
                     title='Average number of phrases per paper in each year range')
    plt.xlabel('Average number of phrases identified by AutoPhrase per paper')
    plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0.2)
    fn = '/bar_avg_phrases_identified.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Histogram of phrases identified
    phrases = out[out['Num Phrases'] < 200]['Num Phrases'] # removes outliers
    phrases.plot.hist(bins=50, title='Distribution of number of phrases per paper')
    plt.xlabel('Number of phrases identified by AutoPhrase per paper')
    plt.subplots_adjust(left=0.2)
    fn = '/hist_phrases_identified.png'
    plt.savefig(outfolder + fn)
    plt.clf()
    return
