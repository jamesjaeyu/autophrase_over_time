# Introduction
Phrase mining is the process of utilizing automated programs for extracting important and high-quality phrases from bodies of text. These phrases can be used in a variety of ways, from extracting major ideas from customer reviews or key points from a scientific paper. However, phrase mining has historically been done with complicated linguistic analyzers trained on specific data, meaning that it is difficult to expand to a larger scope without significant additional human effort. As a way to mine phrases in an expandable way, in any language or domain, AutoPhrase was created. With AutoPhrase, it is possible to input any text corpora without the need for human labels, allowing for much faster extraction of phrases in a variety of documents.

With that in mind, we utilized AutoPhrase to extract the phrases from a database of 3,079,007 computer science research papers aggregated from 1950 to 2017. With this, we can trace the evolution of key ideas through the history of computer science, as well as find which ideas were most common in what years. Additionally, we used the extracted phrases as data to construct a classification model for finding what year a paper belongs to based on its key phrases as a way of showing how strong the connections are between ideas and time.

---

# Data
- [DBLP Citation Network Dataset v10](https://www.aminer.org/citation) (4.08 GB)
    - Four .json files containing information on 3,097,007 papers
- [arXiv Computer Science papers 1992-2017](https://www.kaggle.com/neelshah18/arxivdataset) (72 MB)
    - One .json file containing information on 31,000+ papers

### Exploring the data
The original DBLP V10 data set we utilized for our project contains 3,097,007 total papers. However, during data processing, 530,394 were filtered out due to empty abstracts, as well as 82 others due to our year cut off. That leave us with 2,548,531 papers ranging from 1968 to 2017.

![Papers years](/autophrase_over_time/assets/papers_per_year.png)

We also wanted to use some data from the ArXiv data set for data exploration and later use. Unfortunately, we weren't able to use it for modelling due to its small size, but we still did data exploration on this set.


### Data preparation
Brief description of how the data was processed for further analysis

---

# Methods
### Running AutoPhrase
AutoPhrase has two functions that can be run. Phrase mining and phrasal segmentation. In order to perform the phrase mining step, a single .txt file is required. Once that step is ran, a segmentation model is generated that can be used to segment any other .txt file, including the same input .txt file.

---

# Results
### Phrase mining results
To run AutoPhrase on our data, first we just split it by year to aggregate the most important phrases for each year. In doing this, we found that there was a minimum file size of about 200-300 kB for running AutoPhrase, meaning that we were unable to utilize the years 1950-67. However, we also found that we could group the years together both to enable more years in our data as well as to find salient phrases during longer time periods. Having each year separate makes the data much more sparse, so grouping will enable stronger classification results in the future.

![Phrase mining example results](/autophrase_over_time/assets/autophrase.png)

Phrase mining returns a .txt file containing each extracted phrase, along with their associated phrase quality. Phrase quality ranges from 0.0 to 1.0, with 1.0 being the highest quality.

![Bar chart of number of phrases identified by AutoPhrase](/autophrase_over_time/assets/bar_avg_phrases_identified.png)

Here, we can see that the number of phrases per paper changes drastically depending on the year range. This can be due to factors like average length of input papers for that year range, but could also be dependent upon the range of phrases displayed within that range. A year range with more phrase variety could have less phrases show up per paper due to the lower average scores of the phrases causing them to be excluded from our high-quality phrase list.

![Histogram of number of phrases identified by AutoPhrase](/autophrase_over_time/assets/hist_phrases_identified.png)

Overall, the number of phrases in a paper can vary widely, but the vast majority lie between 15 and 50 phrases.

### Consolidating phrase mining results
Add tables + explanations of the process

### Phrasal segmentation results
When running phrasal segmentation on the input data, we are able to obtain all of the phrases mined by AutoPhrase on a per-paper level. 

![Phrasal segmentation example results](/autophrase_over_time/assets/phrasal_seg.png)

The input text data will mark any identified phrases with phrase markers. The phrase markers and phrases are highlighted in this screenshot for clarity.

### Network visualization
[Link to high-resolution, zoomable image](https://srv2.zoomable.ca/viewer.php?i=img738fab87ba012f22_2)

![Network visualization](/autophrase_over_time/assets/network.png)

The network visualization shows the relationship between phrases in the entire DBLP v10 dataset. More frequent phrases are represented by larger nodes. Nodes that have stronger and more frequent connections with each other will be represented by the same color, determined by their modularity.

---

# Model
### Creating and Refining the Model
Description of the goal of the model and the training data.

### Model Results and Output
Description of the model results on test data along with any relevant tables/graphs

---

# Conclusion
Brief summary of methods & results 

### Future Work & Extensions
Discussion of future work and potential extensions to the project
