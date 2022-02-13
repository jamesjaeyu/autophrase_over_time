"""
DSC180B
Q2 Project
EDA for DBLP v10 and arXiv datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_figures(outfolder):
    """
    Main function for outputting all figures

    >>> generate_figures('../results/figures')
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    v10_papers_per_year('../data/dblp-v10/dblp-v10-counts.csv', outfolder)
    # TODO: Add function calls for any new functions here
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

# TODO: Add functions for arXiv graphs

