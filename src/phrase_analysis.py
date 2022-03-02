"""
DSC180B
Q2 Project
Phrase analysis + visualizations of segmentation results
"""

# TODO: Determine best threshold values for a comprehensive (but uncluttered) graph
# TODO: Associate year ranges with phrases (needs to be for both node and edge data)
#       Can maybe add this as a column in NodeData, but we would also need to know it in EdgeData to see how nodes conect
#       Possibly - add Year Range as a column, but also add it to each phrase?
# TODO: Remove any nodes from NodeData that aren't included in the edges (makes it so node_thresh is no longer needed)

import pandas as pd
import os
from glob import glob
import csv
import altair as alt

BAD_PHRASES_MULTI = set(['an adaptive', 'based approach', 'de los', 'en la',
                         'de la', 'en el', 'de las', '2005 copyright spie',
                         'outcomes of dagstuhl seminar', 'periodicals inc comput appl eng educ',
                         'de donnees', 'de ces', '2008 copyright spie', 'que la',
                         'sur des', 'et les', 'sur la', 'sur les', 'dans les',
                         'proposed algorithm', 'proposed approach', 'proposed method',
                         'a case study', 'recent years', 'an adaptive', 'an overview',
                         'proposed scheme', 'case study', 'obtainable from cpc program library queens',
                         'university belfast n irelandrnrnlicensing provisions',
                         'format targzrnrnprogramming language',
                         'summaryrunprogram title',
                         'distributed program including test data etc'])
BAD_PHRASES_SINGLE = set(['as', 'first', 'most', 'finally', 'e', 'do', 'ii',
                          'n', 'i', 'al', 'k', 'm', 'c', 'd', 'most'])


def gephi_preprocess(infolder, outfolder, edge_thresh):
    """
    Preprocessing of segmentation results for Gephi graph visualization.
    Outputs NodeData.csv and EdgeData.csv to outfolder

    infolder: Folder path containing Gephi phrasal segmentation results
        (see process_seg in model_generation.py)
    outfolder: Folder path for output files
    edge_thresh: Minimum number of overlaps between phrases to be included in the output

    >>> gephi_preprocess('../results/gephi', '../results/temp', 150)
    >>> gephi_preprocess('../results/gephi-all', '../results/temp', 300)
    """
    # Gephi segmentation csvs only contain high-quality, multi-word phrases (no duplicates per paper)
    subfolders = glob(infolder + '/*.csv')
    subfolders = list(filter(lambda x: 'segmented' in x, subfolders))
    seg = pd.DataFrame(columns=['Phrases', 'Year Range'])
    for fp in subfolders:
        df = pd.read_csv(fp, index_col=0)
        df = df.dropna()
        seg = seg.append(df, ignore_index=True)
    seg = seg.dropna()
    seg['Phrases'] = seg['Phrases'].map(lambda x: x.split(','))

    # Removes any papers (rows) with only a single phrase - no edges are possible
    seg = seg[seg.apply(lambda x: len(x['Phrases']) > 1, axis=1)]

    # Creates and outputs EdgeData.csv
    edge_counts = {}
    def get_edges(phrase_lst):
        """
        Helper function to process segmentation results csv to get edge data
        Modifies the edge_dict dictionary
        """
        for phrase in phrase_lst:
            for inner_phrase in phrase_lst:
                # Prevents any bad phrases from being included
                if phrase in BAD_PHRASES_MULTI or inner_phrase in BAD_PHRASES_MULTI:
                    continue

                # Prevents any self-comparisons
                if phrase == inner_phrase: continue

                # Stops any comparisons of existing phrase A - phrase B comparisons
                # We don't need to add the phrase B - phrase A data to the dictionary
                if inner_phrase in edge_counts and phrase in edge_counts[inner_phrase]:
                    continue

                # Creates inner dictionary and adds to count
                if phrase not in edge_counts:
                    edge_counts[phrase] = {}
                if inner_phrase not in edge_counts[phrase]:
                    edge_counts[phrase][inner_phrase] = 0
                edge_counts[phrase][inner_phrase] += 1
        return
    # Applies helper function to seg dataframe. The function will just modify
    # the edge_dict dictionary
    _ = seg.apply(lambda x: get_edges(x['Phrases']), axis=1)
    # Filters out edges that have less than edge_thresh overlaps
    edge_filtered = {}
    edge_phrases = set() # Keeps track of phrases included in EdgeData
    for phrase, phrase_counts in edge_counts.items():
        for inner_phrase, count in phrase_counts.items():
            # Skips any edges with less overlaps than the threshold
            if count < edge_thresh: continue
            # Otherwise, add the edge to the dictionary
            if phrase not in edge_filtered:
                edge_filtered[phrase] = {}
            edge_filtered[phrase][inner_phrase] = count
            edge_phrases.add(phrase)
            edge_phrases.add(inner_phrase)
    # Outputs to EdgeData.csv
    outpath_edge = outfolder + '/EdgeData.csv'
    with open(outpath_edge, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Source', 'Target', 'Weight'])
        for phrase, phrase_counts in edge_filtered.items():
            for inner_phrase, count in phrase_counts.items():
                writer.writerow([phrase, inner_phrase, count])

    # Creates and outputs NodeData.csv
    label_counts = {}
    def get_label_counts(x):
        """
        Helper function to process segmentation results csv to get node counts
        Modifies the label_counts dictionary
        """
        for phrase in x:
            # Skips the phrase if it is not included in EdgeData.csv
            if phrase not in edge_phrases: continue

            # Prevents phrase from being included if it is a bad phrase
            if phrase in BAD_PHRASES_MULTI:
                continue

            # Otherwise add it to the dict
            if phrase not in label_counts:
                label_counts[phrase] = 0
            label_counts[phrase] += 1
        return
    _ = seg.apply(lambda x: get_label_counts(x['Phrases']), axis=1)
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    labels = pd.DataFrame.from_dict(label_counts,
                                    orient='index',
                                    columns=['Count']
                                    ).reset_index().rename(columns={'index': 'ID'})
    labels['Label'] = labels['ID']
    labels = labels[['ID', 'Label', 'Count']]
    # Outputs NodeData.csv
    outpath_node = outfolder + '/NodeData.csv'
    labels.to_csv(outpath_node)


def gephi_preprocess_yearly(infolder, outfolder, edge_thresh):
    """
    Preprocessing segmentation results for Gephi visualizations, but accounting
    for the year range for each phrase

    # NOTE: May need to adjust threshold as edges are only yearly
    # TODO: Different threshold values for each year range? Less papers means less opportunities for connections
    >>> gephi_preprocess_yearly('../results/gephi', '../results/temp', 150)
    """
    # Gephi segmentation csvs only contain high-quality, multi-word phrases (no duplicates per paper)
    subfolders = glob(infolder + '/*.csv')
    subfolders = list(filter(lambda x: 'segmented' in x, subfolders))
    seg = pd.DataFrame(columns=['Phrases', 'Year Range'])
    for fp in subfolders:
        df = pd.read_csv(fp, index_col=0)
        df = df.dropna()
        seg = seg.append(df, ignore_index=True)
    seg = seg.dropna()
    seg['Phrases'] = seg['Phrases'].map(lambda x: x.split(','))

    # Removes any papers (rows) with only a single phrase - no edges are possible
    seg = seg[seg.apply(lambda x: len(x['Phrases']) > 1, axis=1)]

    # Creates and outputs EdgeData.csv
    edge_counts = {}
    def get_edges(line):
        """
        Helper function to process segmentation results csv to get edge data
        Modifies the edge_dict dictionary
        """
        year_range = line['Year Range']
        phrase_lst = line['Phrases']
        for phrase in phrase_lst:
            for inner_phrase in phrase_lst:
                # Prevents any bad phrases from being included
                if phrase in BAD_PHRASES_MULTI or inner_phrase in BAD_PHRASES_MULTI:
                    continue

                # Prevents any self-comparisons
                if phrase == inner_phrase: continue

                # Modifies phrase to include Year Range in parenthesis
                temp_phrase = phrase +  ' (' + year_range + ')'
                temp_inner_phrase = inner_phrase + ' (' + year_range + ')'

                # Stops any comparisons of existing phrase A - phrase B comparisons
                # We don't need to add the phrase B - phrase A data to the dictionary
                if temp_inner_phrase in edge_counts and temp_phrase in edge_counts[temp_inner_phrase]:
                    continue

                # Creates inner dictionary and adds to count
                if temp_phrase not in edge_counts:
                    edge_counts[temp_phrase] = {}
                if temp_inner_phrase not in edge_counts[temp_phrase]:
                    edge_counts[temp_phrase][temp_inner_phrase] = 0
                edge_counts[temp_phrase][temp_inner_phrase] += 1
        return
    # Applies helper function to seg dataframe. The function will just modify
    # the edge_dict dictionary
    _ = seg.apply(lambda x: get_edges(x), axis=1)
    # Filters out edges that have less than edge_thresh overlaps
    edge_filtered = {}
    edge_phrases = set() # Keeps track of phrases included in EdgeData
    for phrase, phrase_counts in edge_counts.items():
        for inner_phrase, count in phrase_counts.items():
            # Skips any edges with less overlaps than the threshold
            if count < edge_thresh: continue
            # Otherwise, add the edge to the dictionary
            if phrase not in edge_filtered:
                edge_filtered[phrase] = {}
            edge_filtered[phrase][inner_phrase] = count
            edge_phrases.add(phrase)
            edge_phrases.add(inner_phrase)
    # Outputs to EdgeData.csv
    outpath_edge = outfolder + '/EdgeDataYearly.csv'
    with open(outpath_edge, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Source', 'Target', 'Weight'])
        for phrase, phrase_counts in edge_filtered.items():
            for inner_phrase, count in phrase_counts.items():
                writer.writerow([phrase, inner_phrase, count])


    # Creates and outputs NodeData.csv
    label_counts = {}
    def get_label_counts(line):
        """
        Helper function to process segmentation results csv to get node counts
        Modifies the label_counts dictionary
        """
        year_range = line['Year Range']
        phrase_lst = line['Phrases']
        for phrase in phrase_lst:
            # Prevents phrase from being included if it is a bad phrase
            if phrase in BAD_PHRASES_MULTI:
                continue

            temp_phrase = phrase +  ' (' + year_range + ')'
            # Skips the phrase if it is not included in EdgeData.csv
            if temp_phrase not in edge_phrases: continue

            # Otherwise add it to the dict
            if temp_phrase not in label_counts:
                label_counts[temp_phrase] = 0
            label_counts[temp_phrase] += 1
        return
    _ = seg.apply(lambda x: get_label_counts(x), axis=1)
    label_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    labels = pd.DataFrame.from_dict(label_counts,
                                    orient='index',
                                    columns=['Count']
                                    ).reset_index().rename(columns={'index': 'ID'})
    labels['Label'] = labels['ID']
    labels = labels[['ID', 'Label', 'Count']]
    # Adds Year Range column
    labels['Year Range'] = labels['Label'].apply(lambda x: x.split()[-1][1:-1])
    # Outputs NodeData.csv
    outpath_node = outfolder + '/NodeDataYearly.csv'
    labels.to_csv(outpath_node)


def gephi_instructions():
    """
    1. Install Gephi at https://gephi.org/

    2. Obtain Node and Edge data (as .csv files)
    - NodeData.csv columns: ID, Label, Count
        (ID and Label are the same - a phrase)
        (Count is the count of that phrase in the phrasal segmentation results)
    - EdgeData.csv columns: Source, Target, Weight
        (The graph is undirected, so the order of Source-Target doesn't matter
         i.e. PhraseA-PhraseB is the same as PhraseB-PhraseA)
        (Source and Target should be different phrases)
        (Weight is the amount of overlaps they have with each other)

    3. Create new Gephi Project (top left: File -> New Project)

    4. Import Node and Edge data to Gephi project
     4a. Go to Data Laboratory (top left)
     4b. Click 'Import Spreadsheet' in the Data Table window
     4c. Select EdgeData.csv
     4d. Use default General CSV options -> click Next
     4e. Make Weight category type 'Double' -> click Finish
     4f. Change Graph Type to 'Undirected'
     4g. Change 'New workspace' to 'Append to existing workspace' -> click Ok
     4h. Repeat 4b-4g for NodeData.csv but for 4d, import as 'Nodes table'

    5. Go to Overview (top left) to view the preliminary graph.

    6. Create Partitions for the nodes (color-coding)
     6a. Go to Statistics menu (right side of window when Overview is open)
     6b. Under Network Overview, go to Modularity
     6c. Click 'Run'
     6d. Adjust Resolution as needed
     6e. Click 'Ok' then 'Close'
     6f. Go to Appearance -> Nodes -> Partition menu (left side of window when Overview is open)
     6g. 'Choose an attribute' -> 'Modularity Class' -> 'Apply'

    7. Add text labels for the nodes in the graph
     7a. Click on the icon in the bottom right of the Graph window
     7b. Click Labels -> Select box for Node
     7c. Adjust size accordingly (easy way to adjust font size is by using the 'Size' box)

    8. Remove any unconnected nodes from the graph
     8a. Go to Filter menu (right side of window when Overview is open)
     8b. Library -> Topology -> Degree Range (double click to add filter)
     8c. Adjust 'Degree Range Settings' in bottom right
     8d. Changing the lower threshold from 0 -> 1+ will remove nodes without any connections
     8e. Select 'Filter' button in bottom right to apply to graph

    9. Change layout of graph
     9a. Go to Layout menu (bottom left side of window when Overview is open)
     9b. Select one of the options and click 'Run'
    (This still needs to be tuned, but right now, Fruchterman Reingold and
     Yifan Hu Proportional seem to be good)
    (The layout of nodes can be adjusted manually as well)
    """
    return
