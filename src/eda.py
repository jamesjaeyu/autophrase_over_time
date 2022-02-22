"""
DSC180B
Q2 Project
EDA for DBLP v10 and arXiv datasets
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import ast

def generate_figures(outfolder):
    """
    Main function for outputting all figures

    >>> generate_figures('../results/figures')
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    v10_papers_per_year('data/dblp-v10/dblp-v10-counts.csv', outfolder)
    arxiv_figures('data/arxiv/arxivData.json', outfolder+'/arxiv_figs')
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


def arxiv_figures(fp, outfolder):
    """
    Takes in filepath for the Arxiv .json data file,
    as well as outfolder for graphs generated
    """
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    arx = pd.read_json(fp)
    arx['tag'] = arx['tag'].apply(ast.literal_eval)

    # documents per year figure
    doc_per_year = arx['year'].value_counts()
    plt.xlabel('year')
    plt.ylabel('count')
    plt.title('Arxiv Documents Per Year')
    plt.hist(arx['year'])
    fn = '/arxiv_papers_per_year.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Average tags per year figure
    arx_taginfo = arx.loc[:, ('year', 'tag')]
    arx_taginfo['num_tags'] = arx_taginfo['tag'].apply(lambda x: len(x))
    arx_tagpy = arx_taginfo.groupby('year').mean()['num_tags']
    plt.xlabel('year')
    plt.ylabel('average tags per document')
    plt.title('Arxiv Average Tags Per Year')
    plt.plot(arx_tagpy)
    fn = '/arxiv_ave_tags_per_year.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Unique tags per year figure
    arx_expandtag = arx_taginfo.explode('tag')
    arx_expandtag['tag'] = arx_expandtag['tag'].apply(lambda x: x['term'])
    arx_extagpy = arx_expandtag.groupby('year').agg({
        'tag': lambda x: len(x.unique())
    })
    plt.title('Number of Unique Tags Per Year')
    plt.xlabel('year')
    plt.ylabel('unique tag amount')
    plt.plot(arx_extagpy)
    fn = '/arxiv_unq_tags_per_year.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Title word count figure
    arx_tinfo = arx.loc[:, ('year', 'title')]
    arx_tinfo['word count'] = arx_tinfo['title'].str.split().apply(len)
    arx_tpy = arx_tinfo.groupby('year').agg({
        'title': ' '.join,
        'word count': 'mean'
    })
    plt.xlabel('year')
    plt.ylabel('average title word count')
    plt.title('Arxiv Title Word Count Per Year')
    plt.plot(arx_tpy.index, arx_tpy['word count'])
    fn = '/arxiv_title_wc_per_year.png'
    plt.savefig(outfolder + fn)
    plt.clf()

    # Abstract word count figure
    arx_abinfo = arx.loc[:, ('year', 'summary')]
    arx_abinfo['word count'] = arx_abinfo['summary'].str.split().apply(len)
    arx_abpy = arx_abinfo.groupby('year').agg({
        'summary': ' '.join,
        'word count': 'mean'
    })
    plt.xlabel('year')
    plt.ylabel('average title word count')
    plt.title('Arxiv Abstract Word Count Per Year')
    plt.plot(arx_abpy.index, arx_abpy['word count'])
    fn = '/arxiv_ab_wc_per_year.png'
    plt.savefig(outfolder + fn)
    plt.clf()
