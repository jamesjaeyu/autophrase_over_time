"""
DSC180B
Q2 Project
Using AutoPhrase on Computer Science papers over time
"""
import os, glob
import pandas as pd
import random
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def obtain_phrases(fp, score_thresh, underscore=False):
    """
    Obtains all phrases in a file above a certain score threshold (exclusive)
    underscore determines if multi-word phrases have underscores between words

    Returns a list of (score, phrase) tuples
    """
    phrases_above_thresh = []
    sym = ' '
    if underscore:
        sym = '_'

    with open(fp, 'r') as f:
        for line in f:
            components = line.split()
            score = float(components[0])
            phrase = sym.join(components[1:])
            if score > score_thresh:
                phrases_above_thresh.append((score, phrase))
    return phrases_above_thresh


def process_seg(fp):
    """
    Given a segmentation.txt filepath, process the phrases so it is ready to
    train on a word2vec model
    (get rid of phrase markers and add underscores to multi-word phrases)

    Returns a 2D-list of the tokenized words in each sentence in the file
    """
    data = []
    with open(fp, 'r', encoding='utf8') as f:
        for line in f:
            line = line.lower()
            out = ''
            while line.find('<phrase>') != -1:
                start_idx = line.find('<phrase>')
                end_idx = line.find('</phrase>')
                out += line[:start_idx]
                words = '_'.join(line[start_idx+8:end_idx].split())
                out += words
                line = line[end_idx+9:]
            out += line
            data.append(word_tokenize(out))


def create_model(data):
    """
    Returns word2vec model trained on segmentation data. The raw data needs
    to be processed by the process_seg function first.
    """
    return Word2Vec(data, min_count = 1, window = 5)


def find_most_similar(fp, score_thresh, model, phrase, num_out):
    """
    fp determines the filepath of the phrases we want to evaluate similarity on.
    score_thresh determines the threshold of scores for phrases we want to evaluate.
    Using a word2vec model, return the num_out most similar phrases to phrase.

    Returns a list of (similarity_score, (phrase quality, phrase)) tuples
    """
    phrases = obtain_phrases(fp, score_thresh, True)

    # Finding num_out most similar phrases
    sim = []
    for x in phrases:
        try:
            score = model.wv.similarity(phrase, x[1])
            sim.append((score, x))
        except:
            continue

    sim.sort(reverse=True)
    return sim[:num_out+1]