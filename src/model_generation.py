"""
DSC180B
Q2 Project
Processing DBLP v10
"""
import pandas as pd
import numpy as np
import time
import sys
import datetime
import pickle
import re
import os
import nltk
import sklearn

from glob import glob
import Levenshtein as Lv # Used to calculate string similarity
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer 
import warnings
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from joblib import dump, load
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")

def obtain_phrases(infolder, threshold=(0.8,0.5)):
    """
    Outputs a csv containing consolidated AutoPhrase results.
    Columns: Phrase Quality, Phrase, Year, Num Words
    (Takes 8 seconds to run)

    infolder: Folder path containing AutoPhrase results by year
    threshold: Tuple containing floats for (single-word, multi-word) thresholds
               for phrase quality

    >>> obtain_phrases('../results/dblp-v10', (0.8, 0.5))
    >>> obtain_phrases('../results/dblp-v10-grouped', (0.8, 0.5))
    """
    start = time.time()
    # Gathers filepaths of each AutoPhrase.txt file
    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/AutoPhrase.txt')

    # Output dataframe
    out_df = pd.DataFrame(columns=['Phrase Quality', 'Phrase', 'Year', 'Num Words'])
    # Processes each AutoPhrase.txt file
    for fp in filepaths:
        year = fp.split('/')[-2]
        df = pd.read_csv(fp, sep='\t', header=None, names=['Phrase Quality', 'Phrase'])
        df = df.dropna()
        df['Year'] = [year] * len(df)
        # Number of words in the phrase
        df['Num Words'] = df['Phrase'].map(str.split).map(len)
        # Filters out single and multi-word phrases based on phrase quality
        filter_qual = lambda x: True if \
            (x['Num Words'] > 1 and x['Phrase Quality'] >= threshold[1]) \
            or (x['Num Words'] == 1 and x['Phrase Quality'] >= threshold[0]) \
            else False
        # Only keeps phrases above given threshold
        valid_idx = df.apply(filter_qual, axis=1)
        df = df[valid_idx]
        # Appends rows to output dataframe
        out_df = out_df.append(df, ignore_index=True)
    # Outputs to infolder as phrases.csv
    outpath = infolder + '/phrases.csv'
    out_df.to_csv(outpath)
    end = time.time()
    return end - start


def process_seg(infolder, outfolder, phrases_fp=None, method='general'):
    """
    Processes phrasal segmentation .txt results
    Outputs .csv files with columns: Phrases, Year Range
    Each row represents a single paper

    method='general': All phrases from phrasal segmentation results are included in output
        phrases_fp can be a random input - it will not be used
    method='model': Only high-quality phrases, no duplicates in each row
    method='gephi': Only multi-word, high-quality phrases, no duplicates in each row

    >>> process_seg('../results/dblp-v10-grouped', '../results/dblp-v10-grouped')

    >>> process_seg('../results/dblp-v10-grouped', \
                    '../results/dblp-v10-model', \
                    '../results/dblp-v10-grouped/phrases.csv', \
                    'model')

    >>> process_seg('../results/dblp-v10-grouped', \
                    '../results/dblp-v10-gephi', \
                    '../results/dblp-v10-grouped/phrases.csv', \
                    'gephi')
    """
    start = time.time()
    if method not in ['general', 'model', 'gephi']:
        print('Choose a valid process_seg method: general, model, gephi')
        return

    # Only needed if the method is 'model' or 'gephi'
    if method != 'general':
        # Reading in AutoPhrase results csv
        df_phrases = pd.read_csv(phrases_fp, index_col=0)
        # Set containing all the high-quality phrases from AutoPhrase results
        # (single-words with quality >= 0.8, multi >= 0.5)
        unique_phrases = set(df_phrases['Phrase'].values)

    # Getting filepaths for segmentation.txt files
    infolder = '../results/dblp-v10-grouped'
    subfolders = glob(infolder + '/*/')
    subfolders = [x.split('\\')[1] for x in subfolders]
    filepaths = []
    for sub in subfolders:
        filepaths.append(infolder + '/' + sub + '/segmentation.txt')

    def extract_phrases(line):
        """
        Helper function for processing phrasal segmentation .txt results
        Outputs a string containing phrases in line, separated by commas
        """
        line = line.lower()
        if method == 'general':
            out = []
        else:
            out = set()
        # Processes the line until there are no phrases left
        while line.find('<phrase>') != -1:
            start_idx = line.find('<phrase>')
            end_idx = line.find('</phrase>')
            # Obtains text between phrase markers
            phrase = line[start_idx+8:end_idx]
            line = line[end_idx+9:]
            # Removes any non-alphanumeric characters
            phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
            phrase = re.sub(r'-', ' ', phrase)
            # Adds phrase to output
            if method == 'general':
                out.append(phrase)
            else:
                # Ensures we only keep multi-word phrases for 'gephi' method
                # if method == 'gephi':
                #     num_words = len(phrase.split())
                #     if num_words == 1:
                #         continue

                # If the phrase is contained within the AutoPhrase results
                # it is a high-quality phrase, so we can add it
                if phrase in unique_phrases:
                    out.add(phrase)
        # Output will be all the phrases in a single string, separated by commas
        if method != 'general':
            out = list(out)
        out = ','.join(out)
        return out

    # Processing the segmentation.txt files for each year range
    # Outputs into separate .csv files for each year range
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    for fp in filepaths:
        year = fp.split('/')[3]
        df = pd.read_csv(fp, sep='\n', header=None, names=['Phrases'])
        df['Year Range'] = [year] * len(df)
        df['Phrases'] = df.apply(lambda x: extract_phrases(x['Phrases']), axis=1)
        outpath = outfolder + '/' + year + '_segmented_' + method + '.csv'
        df.to_csv(outpath)
    end = time.time()
    return str(end-start) + ' seconds to run'


def phrase_counts(infolder):
    """
    Using the segmentation results, returns a dictionary with the phrase counts
    across each year range.
    Dictionary format will be {Year Range: {phrase: count}}

    >>> phrase_counts('../results/dblp-v10-grouped')
    """
    def add_counts(x):
        """
        Helper function for processing segmented.csv dataframes
        Modifies the counts output dictionary
        """
        phrases = x['Phrases'].split(',')
        year = x['Year']
        for phrase in phrases:
            if phrase not in counts[year]:
                counts[year][phrase] = 0
            counts[year][phrase] += 1

    # Obtains filepaths for all segmented.csv files
    subfolders = glob(infolder + '/*.csv')
    filepaths = list(filter(lambda x: 'segmented' in x, subfolders))

    # Creates dataframe with all segmentation.csv data combined
    seg = pd.DataFrame(columns=['Phrases', 'Year'])
    for fp in filepaths:
        df = pd.read_csv(fp, index_col=0)
        df = df.dropna()
        #df['Num Phrases'] = df.apply(lambda x: len(x['Phrases'].split(',')), axis=1)
        #df = df.drop('Phrases', axis=1)
        seg = seg.append(df, ignore_index=True)

    # Output dictionary
    counts = {}
    for yr in seg['Year'].unique():
        counts[yr] = {}

    # Helper function will modify the counts dictionary
    seg.apply(add_counts, axis=1)

    # Sorts the inner dictionaries of counts in descending order based on frequency
    for key, val in counts.items():
        counts[key] = dict(sorted(val.items(), key=lambda item: item[1], reverse=True))

    # Dictionary that only contains multi-word phrases and counts
    multi_counts = {}
    for year_range, phrase_counts in counts.items():
        multi_counts[year_range] = {key: val for key, val in phrase_counts.items() if len(key.split()) > 1}

    # Creates counts dictionary but with percent as values, rather than raw frequency
    counts_per = {}
    for year_range, phrase_counts in counts.items():
        total_count_yr = sum(phrase_counts.values())
        prop_counts = {}
        for key, val in phrase_counts.items():
            prop_counts[key] = (val / total_count_yr) * 100
        counts_per[year_range] = prop_counts


def find_similar(input_phrase, fp):
    """
    Given an input_phrase, return the most similar phrase from the AutoPhrase
    results, along with all of the years in which that phrase has appeared.

    Output format: (distance, most similar phrase, [years where the MSP appears])

    >>> find_similar('convolutional neural networks', '../results/dblp-v10-phrases-uniquebyyear.csv')
    (0.0, 'convolutional neural networks', [2012, 2013, 2014, 2015, 2016, 2017])
    """
    df = pd.read_csv(fp)
    df['Dist'] = df.apply(lambda x: Lv.distance(input_phrase, x['Phrase'])
                        if isinstance(x['Phrase'], str) else float('inf'), axis=1)
    df_sorted = df.sort_values('Dist')
    closest = df_sorted['Phrase'][0]
    closest_dist = df_sorted['Dist'][0]
    years = list(df[df['Phrase'] == closest]['Year'])
    return (closest_dist, closest, years)


def generate_model(fp):
    """
    fp: Filepath containing AutoPhrase phrase mining results csv

    Features (x):
    -------------
    Phrase (str)
        uses OneHotEncoder. Potential issues can arise when a phrase that
        is not in the training set is passed into the model. Errors resolved by the
        'handle_unknown' parameter in the ohe_pipe
    num_years (int)
        uses StandardScaler. Tells us the number of years a phrase has appeared in
    Phrase Quality (float)
        no modification. May need to normalize in the future. Also, the
        'phrases' dataframe only contains high-quality phrases (multi >= 0.6, single >= 0.8)
    Num Words (int)
        Number of words in the phrase

    Label (y):
    ---------------
    Year (int)
        The year the phrase belongs to.

    Returns
    -------
    float
        Accuracy on the test set

    >>> refined_model('../results/dblp-v10-grouped/phrases.csv')
    """
    df = pd.read_csv(fp, index_col=0)
    df = df.dropna()
    # Creates 'num_years' column - number of year ranges a phrase has appeared in
    phr_counts = df.groupby('Phrase').size()
    df['num_years'] = df.apply(lambda x: phr_counts[x['Phrase']], axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df[['Phrase', 'num_years', 'Phrase Quality']],
                                                        df['Year'])

    # Creates pipeline
    std_pipe = Pipeline([('scale', StandardScaler())])
    ohe_pipe = Pipeline([('one-hot', OneHotEncoder(handle_unknown='ignore'))])
    ct = ColumnTransformer(transformers=[('ohe', ohe_pipe, ['Phrase']),
                                        ('scale', std_pipe, ['num_years']),
                                        ('keep', 'passthrough', ['Phrase Quality', 'Num Words'])
                                        ])
    pl = Pipeline([('transform', ct), ('classifier', AdaBoostClassifier(n_estimators=100,
                                                                        learning_rate=1.1))])

    # Trains model
    pl.fit(X_train, y_train)

    # Baseline accuracy - most popular label
    base_acc = (y_test == '2010-2014').mean()

    return pl.score(X_test, y_test)

#...
# input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
# output: baseline_lr_model_pkl = "results/autophrase/dblp-v10/grouped/baseline_lr_model.pkl"
def baseline_tfidf_model(extract_phrases_per_paper_pkl_file, baseline_lr_model_pkl):
    phrases_by_paper = pd.read_pickle(extract_phrases_per_paper_pkl_file)
    print(phrases_by_paper.groupby('year_bracket_int_encoded').size())
    # 0        333
    # 1        852
    # 2       2642
    # 3       5568
    # 4       9371
    # 5      16844
    # 6      32767
    # 7      71404
    # 8     144387
    # 9     309672
    # 10    651834
    # 11    872491
    # 12    430036

    print(phrases_by_paper.groupby('year_bracket_int_encoded').size()/len(phrases_by_paper))
    # 0     0.000131
    # 1     0.000334
    # 2     0.001037
    # 3     0.002185
    # 4     0.003677
    # 5     0.006610
    # 6     0.012859
    # 7     0.028021
    # 8     0.056662
    # 9     0.121526
    # 10    0.255802
    # 11    0.342395
    # 12    0.168761

    # split tran_set and test_set based on the distribution of the column of 'year_bracket_int_encoded'
    # tran_set and test_set have the same distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(phrases_by_paper, phrases_by_paper["year_bracket_int_encoded"]):
        strat_train_set = phrases_by_paper.loc[train_index]
        strat_test_set = phrases_by_paper.loc[test_index]
    #print(f"len(strat_train_set)={len(strat_train_set)} len(strat_test_set)={len(strat_test_set)} {len(strat_test_set)/len(strat_train_set)}")
    #print(strat_train_set.groupby('year_bracket_int_encoded').size()/len(strat_train_set))
    #print(strat_test_set.groupby('year_bracket_int_encoded').size()/len(strat_test_set))

    X_train, X_test, y_train, y_test = strat_train_set.loc[:, 'phrases'], strat_test_set.loc[:, 'phrases'], strat_train_set.loc[:, 'year_bracket_int_encoded'], strat_test_set.loc[:, 'year_bracket_int_encoded']

    # Convert X_train into a matrix of TF-IDF features.
    start = time.time()
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = {'english'}, max_features=1000000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.values)
    sorted_vocabulary_ = dict( sorted(tfidf_vectorizer.vocabulary_.items(), key=lambda item: item[1], reverse=True))
    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: tfidf_vectorizer.fit_transform {(end - start):.03f} seconds", flush=True)
    #print(sorted_vocabulary_)
    # term frequency: {'zyx': 56126, 'zynq': 56125, 'zynga': 56124, 'zygote': 56123, 'zygomatic': 56122, ...}

    # baseline model
    # One-vs-the-rest (OvR) used for multilabel classification.
    # https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    # a paper can belong to one of the following classes:
    #   '1950-1959'/0,
    #   '1960-1964'/1
    #   '1965-1969'/2
    #   '1970-1974'/3
    #   '1975-1979'/4
    #   '1980-1984'/5
    #   '1985-1989'/6
    #   '1990-1994'/7
    #   '1995-1999'/8
    #   '2000-2004'/9
    #   '2005-2009'/10
    #   '2010-2014'/11
    #   '2015-2017'/12
    start = time.time()
    log_regress = LogisticRegression(solver='lbfgs', max_iter=100)
    clf_name = "LogisticRegression"
    baseline_clf = OneVsRestClassifier(log_regress, n_jobs = 32)
    baseline_clf.fit(X_train_tfidf, y_train)
    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: {clf_name} fit {(end - start):.03f} seconds", flush=True)

    # evaluation of the model
    X_test_tfidf = tfidf_vectorizer.transform(X_test.values)
    y_test_pred = baseline_clf.predict(X_test_tfidf)
    baseline_f1 = f1_score(y_test, y_test_pred, average="micro")
    print(f"baseline_f1={baseline_f1}")
    # baseline_f1=0.7659038421163132

    if os.path.exists(baseline_lr_model_pkl):
        os.remove(baseline_lr_model_pkl)
    with open(baseline_lr_model_pkl,'wb') as baseline_lr_model_pkl_file:
        pickle.dump(baseline_clf, baseline_lr_model_pkl_file)
    print(f"baseline_clf has been saved to {baseline_lr_model_pkl}")

#...
# input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
# output: best_linear_svm_model_pkl = "results/autophrase/dblp-v10/grouped/best_linear_svm_model.pkl"
def optimize_model_parameters(extract_phrases_per_paper_pkl_file, best_linear_svm_model_pkl):
    phrases_by_paper = pd.read_pickle(extract_phrases_per_paper_pkl_file)

    # split tran_set and test_set based on the distribution of the column of 'year_bracket_int_encoded'
    # tran_set and test_set have the same distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(phrases_by_paper, phrases_by_paper["year_bracket_int_encoded"]):
        strat_train_set = phrases_by_paper.loc[train_index]
        strat_test_set = phrases_by_paper.loc[test_index]

    X_train, X_test, y_train, y_test = strat_train_set.loc[:, 'phrases'], strat_test_set.loc[:, 'phrases'], strat_train_set.loc[:, 'year_bracket_int_encoded'], strat_test_set.loc[:, 'year_bracket_int_encoded']

    # Convert X_train into a matrix of TF-IDF features.
    start = time.time()
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = {'english'}, max_features=1000000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.values)
    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: tfidf_vectorizer.fit_transform {(end - start):.03f} seconds", flush=True)

    start = time.time()
    parameters = {
        "estimator__C": [0.1, 10, 15, 20],
    }

    svc = sklearn.svm.LinearSVC(random_state=42)
    clf_name = 'svm.LinearSVC'
    svc_clf = OneVsRestClassifier(svc, n_jobs=32)
    grid_search_clf = GridSearchCV(svc_clf, param_grid=parameters, cv = 3, verbose = 3, scoring = 'f1_micro', refit = True, return_train_score=True)
    grid_search_clf.fit(X_train_tfidf, y_train)
    #   [CV 1/3] END estimator__C=0.1;, score=(train=0.783, test=0.766) total time=   9.0s
    #   [CV 2/3] END estimator__C=0.1;, score=(train=0.783, test=0.765) total time=   9.1s
    #   [CV 3/3] END estimator__C=0.1;, score=(train=0.783, test=0.765) total time=   9.0s
    #   [CV 1/3] END estimator__C=10;, score=(train=0.826, test=0.791) total time= 2.6min
    #   [CV 2/3] END estimator__C=10;, score=(train=0.827, test=0.790) total time= 2.6min
    #   [CV 3/3] END estimator__C=10;, score=(train=0.827, test=0.790) total time= 2.6min
    #   [CV 1/3] END estimator__C=15;, score=(train=0.827, test=0.791) total time= 3.7min
    #   [CV 2/3] END estimator__C=15;, score=(train=0.828, test=0.790) total time= 3.8min
    #   [CV 3/3] END estimator__C=15;, score=(train=0.828, test=0.790) total time= 3.7min
    #   [CV 1/3] END estimator__C=20;, score=(train=0.828, test=0.791) total time= 4.9min
    #   [CV 2/3] END estimator__C=20;, score=(train=0.828, test=0.790) total time= 5.0min
    #   [CV 3/3] END estimator__C=20;, score=(train=0.828, test=0.790) total time= 5.3min

    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: {clf_name} fit {(end - start):.03f} seconds", flush=True)

    print(grid_search_clf.best_params_)
    print(grid_search_clf.best_estimator_)
    cvres = grid_search_clf.cv_results_
    for mean_test_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_test_score, params)
    #   {'estimator__C': 20}
    #   OneVsRestClassifier(estimator=LinearSVC(C=20, random_state=42), n_jobs=32)
    #   0.7654118593516993 {'estimator__C': 0.1}
    #   0.7903441645082804 {'estimator__C': 10}
    #   0.7904589514166863 {'estimator__C': 15}
    #   0.7905183070402636 {'estimator__C': 20}

    # evaluation of the model
    X_test_tfidf = tfidf_vectorizer.transform(X_test.values)
    y_test_pred = grid_search_clf.predict(X_test_tfidf)
    grid_search_clf_f1 = f1_score(y_test, y_test_pred, average="micro")
    print(f"grid_search_clf_f1={grid_search_clf_f1}")
    # grid_search_clf_f1=0.7954226602647746

    if os.path.exists(best_linear_svm_model_pkl):
        os.remove(best_linear_svm_model_pkl)
    with open(best_linear_svm_model_pkl,'wb') as best_linear_svm_model_pkl_file:
        pickle.dump(grid_search_clf, best_linear_svm_model_pkl_file)
    print(f"grid_search_clf has been saved to {best_linear_svm_model_pkl}")

# input: extract_phrases_per_paper_pkl_file = "results/autophrase/dblp-v10/grouped/extract_phrases_per_paper.pkl"
# input: best_linear_svm_model_pkl = "results/autophrase/dblp-v10/grouped/best_linear_svm_model.pkl"
# output: confusion_matrix_png_file = "results/figures/best_linear_svm_model_confusion_matrix.png"
# output: normalized_confusion_matrix_png_file = "results/figures/best_linear_svm_model_normalized_confusion_matrix.png"
def confusion_matrix_analysis(extract_phrases_per_paper_pkl_file, best_linear_svm_model_pkl,
    confusion_matrix_png_file, normalized_confusion_matrix_png_file):
    if not os.path.exists(best_linear_svm_model_pkl):
        print(f"{best_linear_svm_model_pkl} does not exist.")
        sys.exit()

    if not os.path.exists(extract_phrases_per_paper_pkl_file):
        print(f"{extract_phrases_per_paper_pkl_file} does not exist.")
        sys.exit()

    # load best svm LinearSVC model
    with open(best_linear_svm_model_pkl,'rb') as best_linear_svm_model_pkl_file:
        grid_search_clf = pickle.load(best_linear_svm_model_pkl_file)

    phrases_by_paper = pd.read_pickle(extract_phrases_per_paper_pkl_file)

    # split tran_set and test_set based on the distribution of the column of 'year_bracket_int_encoded'
    # tran_set and test_set have the same distribution
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(phrases_by_paper, phrases_by_paper["year_bracket_int_encoded"]):
        strat_train_set = phrases_by_paper.loc[train_index]
        strat_test_set = phrases_by_paper.loc[test_index]

    X_train, X_test, y_train, y_test = strat_train_set.loc[:, 'phrases'], strat_test_set.loc[:, 'phrases'], strat_train_set.loc[:, 'year_bracket_int_encoded'], strat_test_set.loc[:, 'year_bracket_int_encoded']

    # Convert X_train into a matrix of TF-IDF features.
    start = time.time()
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = {'english'}, max_features=1000000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.values)
    end = time.time()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_datetime}: tfidf_vectorizer.fit_transform {(end - start):.03f} seconds", flush=True)

    y_train_pred = grid_search_clf.predict(X_train_tfidf)
    grid_search_clf_train_f1 = f1_score(y_train, y_train_pred, average="micro")
    print(f"grid_search_clf_train_f1={grid_search_clf_train_f1}")
    # grid_search_clf_train_f1=0.8228058040970098

    #   0  ->  1950-1959
    #   1  ->  1960-1964
    #   2  ->  1965-1969
    #   3  ->  1970-1974
    #   4  ->  1975-1979
    #   5  ->  1980-1984
    #   6  ->  1985-1989
    #   7  ->  1990-1994
    #   8  ->  1995-1999
    #   9  ->  2000-2004
    #   10 ->  2005-2009
    #   11 ->  2010-2014
    #   12 ->  2015-2017
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    #plt.show()
    plt.imsave(confusion_matrix_png_file, conf_mx, cmap=plt.cm.gray)

    row_sums = conf_mx.sum(axis=1, keepdims=True)
    normalized_conf_mx = conf_mx / row_sums
    np.fill_diagonal(normalized_conf_mx, 0)

    plt.matshow(normalized_conf_mx, cmap=plt.cm.gray)
    #plt.show()
    plt.imsave(normalized_confusion_matrix_png_file, normalized_conf_mx, cmap=plt.cm.gray)