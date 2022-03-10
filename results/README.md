# Description of results folder


* `dblp-v10-grouped`
    * Contains AutoPhrase results for grouped years in the DBLP-v10 dataset
    * Contains processed phrasal segmentation results in .csv files for each year range
* `figures`
    * Contains output figures from 'eda' target on run.py
* `gephi-data`
    * Contains .csv files for Gephi network visualization
    * NodeData.csv and EdgeData.csv are for the phrase network on the entire dataset
    * NodeDataYearly.csv and EdgeDataYearly.csv are for the phrase network by year range
    * Instructions on how to utilize these files can be found in phrase_analysis.py in the gephi_instructions() function


## For AutoPhrase results folders  
* `AutoPhrase.txt`
    * Contains both single-word and multi-word phrases and phrase qualities
* `AutoPhrase_multi-words.txt`
    * Contains only multi-word phrases and phrase qualities
* `AutoPhrase_single-words.txt`
    * Contains only single-word phrases and phrase qualities
* `language.txt`
    * Contains the detected language of the input .txt file
* `segmentation.model`
    * Contains the trained model information for phrasal segmentation on any .txt file
* `token_mapping.txt`
    * Contains the counts for single-word tokens in the input .txt file
