"""
DSC180B
Q2 Project
Script for running targets
"""

import sys
import json

from src.process_dblp_v10 import download_dblp_v10, download_dblp_v10_using_requests, process_v10, \
    process_v10_txt, autophrase_one_dir, extract_phrases_one_dir, count_phrase_one_dir, \
    extract_phrases_per_paper_one_dir, convert_extract_phrases_per_paper_csv_to_df_pickle
from src.eda import generate_figures
from src.model_generation import obtain_phrases, process_seg, baseline_tfidf_model, optimize_model_parameters, \
    confusion_matrix_analysis
from src.phrase_analysis import phrase_tables

def main(targets):
    """
    Targets: data, eda, model
    """
    # Updates targets to include everything if 'all' is included
    if 'all' in targets:
        targets = ['data', 'eda', 'model', 'analysis']

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

        # analysis
        analysis_cfg = json.load(open('config/analysis-params.json'))
        phrase_tables(analysis_cfg)
    else:
        # Runs each relevant target in targets
        if 'data' in targets:
            data_cfg = json.load(open('config/data-params.json'))

            #download_dblp_v10(data_cfg['dblp_v10_url'])
            #download_dblp_v10_using_requests(data_cfg['dblp_v10_url'])

            ## Processes DBLP v10 dataset into aggregated .txt files by year
            #year_grouping = True
            #output_type = 'txt'
            #year_grouping = False
            # output_type = 'csv'

            #process_v10(data_cfg['in_folder'], data_cfg['out_grouped_txt_folder'], year_grouping, output_type)
            #autophrase_one_dir(data_cfg['autophrase_in_folder'], data_cfg['autophrase_out_folder'])
            #extract_phrases_one_dir(data_cfg['autophrase_out_folder'])
            #count_phrase_one_dir(data_cfg['autophrase_out_folder'])

            #...
            only_first_year_bracket = False
            # input: autophrase_dblp_v10_grouped_folder=results/autophrase/dblp-v10/grouped
            # output: extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
            autophrase_dblp_v10_grouped_folder = data_cfg['autophrase_out_folder']
            extract_phrases_per_paper_csv_file = data_cfg['extract_phrases_per_paper_csv_file']
            extract_phrases_per_paper_one_dir(autophrase_dblp_v10_grouped_folder, extract_phrases_per_paper_csv_file, only_first_year_bracket)

            # input: extract_phrases_per_paper_csv_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.csv
            # output: extract_phrases_per_paper_pkl_file=results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl
            extract_phrases_per_paper_csv_file = data_cfg['extract_phrases_per_paper_csv_file']
            extract_phrases_per_paper_pkl_file = data_cfg['extract_phrases_per_paper_pkl_file']
            convert_extract_phrases_per_paper_csv_to_df_pickle(extract_phrases_per_paper_csv_file, extract_phrases_per_paper_pkl_file)


        if 'eda' in targets:
            eda_cfg = json.load(open('config/eda-params.json'))
            # Generates figures and outputs to outfolder directory
            generate_figures(eda_cfg['outfolder'])

        if 'model' in targets:
            model_cfg = json.load(open('config/model-params.json'))

            #threshold = [float(x) for x in model_cfg['threshold'].split(',')]
            #threshold = (threshold[0], threshold[1])
            #obtain_phrases(model_cfg['infolder'], threshold)
            #obtain_phrases(model_cfg['infolder_grouped'], threshold)
            #baseline_model(model_cfg['fp_grouped'])

            # input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
            # output: baseline_lr_model_pkl = "results/autophrase/dblp-v10/grouped/baseline_lr_model.pkl"
            #baseline_tfidf_model(model_cfg['extract_phrases_per_paper_pkl_file'], model_cfg['baseline_lr_model_pkl'])

            # input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
            # output: best_linear_svm_model_pkl = "results/autophrase/dblp-v10/grouped/best_linear_svm_model.pkl"
            optimize_model_parameters(model_cfg['extract_phrases_per_paper_pkl_file'], model_cfg['best_linear_svm_model_pkl'])
        
        if 'analysis' in targets:
            analysis_cfg = json.load(open('config/analysis-params.json'))
            phrase_tables(analysis_cfg)
            # input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
            # input: best_linear_svm_model_pkl = "results/autophrase/dblp-v10/grouped/best_linear_svm_model.pkl"
            # output: confusion_matrix_png_file = "results/figures/best_linear_svm_model_confusion_matrix.png"
            # output: normalized_confusion_matrix_png_file = "results/figures/best_linear_svm_model_normalized_confusion_matrix.png"
            confusion_matrix_analysis(analysis_cfg['extract_phrases_per_paper_pkl_file'], analysis_cfg['best_linear_svm_model_pkl'],
                analysis_cfg['confusion_matrix_png_file'], analysis_cfg['normalized_confusion_matrix_png_file']) 

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
