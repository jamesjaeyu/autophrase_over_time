"""
DSC180B
Q2 Project
Script for running targets
"""

import sys
import json

from src.process_dblp_v10 import process_v10_txt
from src.eda import generate_figures

def main(targets):
    """
    Targets: data, eda, autophrase, model
    """
    # Updates targets to include everything if 'all' is included
    if 'all' in targets:
        targets = ['data', 'eda', 'autophrase', 'model']

    # Runs each relevant target in targets
    if 'data' in targets:
        data_cfg = json.load(open('config/data-params.json'))
        # Using the cfg, run the relevant functions
        #data = get_data(**data_cfg)

    if 'eda' in targets:
        eda_cfg = json.load(open('config/eda-params.json'))
        generate_figures('../results/figures')

    if 'autophrase' in targets:
        autophrase_cfg = json.load(open('config/autophrase-params.json'))

    if 'model' in targets:
        model_cfg = json.load(open('config/model-params.json'))


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)