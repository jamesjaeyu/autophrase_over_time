# Description of results folder

* arxiv-FULL
    * Contains AutoPhrase results for the entire arXiv dataset (all years combined)
* arxiv 
    * Contains AutoPhrase results for each year in the arxiV dataset (1993-2018)
    * Years 1993-1999, 2001-2003, 2006, 2008 did not have enough training data for AutoPhrase to run properly
* dblp-v10-FULL
    * Contains AutoPhrase results for the entire DBLP-v10 dataset (all years combined)
* dblp-v10
    * Contains AutoPhrase results for each year in the DBLP-v10 dataset (1950-2017)
* dblp-v13/v13_content_2018
    * Contains AutoPhrase results for DBLP-v13's 2018 data
* pictures
    * Contains various pictures related to function outputs, graphs, and errors
* dblp-v10-phrases-unique.csv
    * Contains the consolidated results of AutoPhrase on the DBLP-v10 dataset
    * Phrases are unique overall, meaning only the earliest instance is included
* dblp-v10-phrases-uniquebyyear.csv
    * Contains the consolidated results of AutoPhrase on the DBLP-v10 dataset
    * Phrases are unique by year, so multiple occurrences across years are included


## For AutoPhrase results folders  
* AutoPhrase.txt
    * Contains both single-word and multi-word phrases and phrase qualities
* AutoPhrase_multi-words.txt
    * Contains only multi-word phrases
* AutoPhrase_single-words.txt
    * Contains only single-word phrases
* language.txt
    * Contains the detected language of the input .txt file
* segmentation.model
    * Contains the model information for phrasal segmentation on the input .txt file
* token_mapping.txt
    * Contains the counts for each token in the input .txt file
