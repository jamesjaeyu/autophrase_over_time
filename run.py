"""
DSC180B
Q2 Project
Script for running targets
"""

import sys
import json

from src.process_dblp_v10 import download_dblp_v10, download_dblp_v10_using_requests, process_v10, \
    process_v10_txt, autophrase_one_dir
from src.eda import generate_figures
from src.model_generation import obtain_phrases, process_seg, baseline_model

def main(targets):
    """
    Targets: data, eda, model
    """
    # Updates targets to include everything if 'all' is included
    if 'all' in targets:
        targets = ['data', 'eda', 'model']

    if 'test' in targets:
        # 'test' target will be the same for 'eda' but different for 'data' and 'model'
        # (uses smaller test data)
        targets = ['data', 'eda', 'model']

        # data
        data_cfg = json.load(open('config/data_test-params.json'))
        process_v10_txt(data_cfg['infolder'])

        # eda
        eda_cfg = json.load(open('config/eda-params.json'))
        generate_figures(eda_cfg['outfolder'])

        # model
        model_cfg = json.load(open('config/model_test-params.json'))
        threshold = [float(x) for x in model_cfg['threshold'].split(',')]
        threshold = (threshold[0], threshold[1])
        obtain_phrases(model_cfg['infolder'], threshold)
        process_seg(model_cfg['infolder'])
        baseline_model(model_cfg['fp'])
    else:
        # Runs each relevant target in targets
        if 'data' in targets:
            data_cfg = json.load(open('config/data-params.json'))

            #download_dblp_v10(data_cfg['dblp_v10_url'])
            #download_dblp_v10_using_requests(data_cfg['dblp_v10_url'])

            ## Processes DBLP v10 dataset into aggregated .txt files by year
            year_grouping = True
            output_type = 'txt'
            #year_grouping = False
            # output_type = 'csv'

            #process_v10(data_cfg['in_folder'], data_cfg['out_grouped_txt_folder'], year_grouping, output_type)
            autophrase_one_dir(data_cfg['autophrase_in_folder'], data_cfg['autophrase_out_folder'])

        if 'eda' in targets:
            eda_cfg = json.load(open('config/eda-params.json'))
            # Generates figures and outputs to outfolder directory
            generate_figures(eda_cfg['outfolder'])

        if 'model' in targets:
            model_cfg = json.load(open('config/model-params.json'))

            threshold = [float(x) for x in model_cfg['threshold'].split(',')]
            threshold = (threshold[0], threshold[1])
            #obtain_phrases(model_cfg['infolder'], threshold)
            obtain_phrases(model_cfg['infolder_grouped'], threshold)
            baseline_model(model_cfg['fp_grouped'])

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
