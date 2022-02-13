"""
DSC180B
Q2 Project
Script for running targets
"""

import sys
import json

from src.process_dblp_v10 import process_v10_txt
from src.eda import generate_figures
from src.model_generation import obtain_phrases
# TODO: Import relevant files/functions for running targets

def main(targets):
    """
    Targets: data, eda, model
    """
    # Updates targets to include everything if 'all' is included
    if 'all' in targets:
        targets = ['data', 'eda', 'autophrase', 'model']

    # Runs each relevant target in targets
    if 'data' in targets:
        data_cfg = json.load(open('config/data-params.json'))
        # TODO: Add function for initial download of dataset

        # Processes DBLP v10 dataset into aggregated .txt files by year
        process_v10_txt(data_cfg['infolder'])

    if 'eda' in targets:
        eda_cfg = json.load(open('config/eda-params.json'))
        # Generates figures and outputs to outfolder directory
        generate_figures(eda_cfg['outfolder'])

    if 'model' in targets:
        model_cfg = json.load(open('config/model-params.json'))
        # Processes the AutoPhrase results in the results/dblp-v10 folder
        obtain_phrases(model_cfg['infolder'], True)
        obtain_phrases(model_cfg['infolder'], False)
        # TODO: Add any new functions related to the model
        

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
