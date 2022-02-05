# Description of src folder

* model_generation.py
    * Consolidates phrases to output .csv files, perform phrase similarity search, and generate model
* process_arxiv.py
    * Processes arXiv dataset and outputs to yearly aggregate .txt and .csv files, and an aggregate .txt with all papers combined.
* process_dblp_v10.py
    * Processes DBLP-v10 dataset and outputs to yearly aggregate .txt and .csv files, and an aggregate .txt with all papers combined.
* process_dblp_v13.py
    * Attempts to process DBLP-v13 dataset and output, but has issues with many papers having json.loads() formatting errors.
* run.py
    * Run file for Dockerimage
