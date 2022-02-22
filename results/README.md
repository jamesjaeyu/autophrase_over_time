# Description of results folder

Note: Some files within this folder require Git LFS (Large File Storage). You can install it [here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)

* `arxiv`
    * Contains AutoPhrase results for each year in the arxiV dataset (1993-2018)
    * Years 1993-1999, 2001-2003, 2006, 2008 did not have enough training data for AutoPhrase to run properly
    * Also contains AutoPhrase results on all years combined (`ALL_YEARS` folder)
* `dblp-v10-grouped`
    * Contains AutoPhrase results for grouped years in the DBLP-v10 dataset
    * Contains processed phrasal segmentation results in .csv files for each year range
* `dblp-v10`
    * Contains AutoPhrase results for each year in the DBLP-v10 dataset (1950-2017)
    * Also contains AutoPhrase results on all years combined (`ALL_YEARS` folder)
* `figures`
    * Contains output figures from 'eda' target on run.py


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
