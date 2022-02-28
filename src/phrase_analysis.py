"""
DSC180B
Q2 Project
Phrase analysis + visualizations of segmentation results
"""

import pandas as pd
import os
from glob import glob
import csv
import altair as alt


def gephi_preprocess(infolder, outfolder, node_thresh, edge_thresh):
    """
    Preprocessing of segmentation results for Gephi graph visualization

    >>> gephi_preprocess('../results/gephi', '../results, 300, 300)
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

    # Removes any papers with only a single phrase
    seg = seg[seg.apply(lambda x: len(x['Phrases']) > 1, axis=1)]

    # Creates and outputs NodeData.csv
    label_counts = {}
    def get_label_counts(x):
        """
        Helper function to process segmentation results csv to get node counts
        """
        for phrase in x:
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
    # Only keeps nodes that have counts above node_thresh
    labels = labels[labels['Count'] > node_thresh]
    # Outputs NodeData.csv
    outpath_node = outfolder + '/NodeData.csv'
    labels.to_csv(outpath_node)

    # Creates and outputs EdgeData.csv
    edge_dict = {}
    def get_edges(phrase_lst):
        """
        Helper function to process segmentation results csv to get edge data
        """
        for phrase in phrase_lst:
            for inner_phrase in phrase_lst:
                # Prevents any self-comparisons
                if phrase == inner_phrase: continue

                # Stops any comparisons of existing phrase A - phrase B comparisons
                # We don't need to add the phrase B - phrase A data to the dictionary
                if inner_phrase in edge_dict and phrase in edge_dict[inner_phrase]:
                    continue

                # Creates inner dictionary and adds to count
                if phrase not in edge_dict:
                    edge_dict[phrase] = {}
                if inner_phrase not in edge_dict[phrase]:
                    edge_dict[phrase][inner_phrase] = 0
                edge_dict[phrase][inner_phrase] += 1
        return
    # Applies helper function to seg dataframe. The function will just modify
    # the edge_dict dictionary
    _ = seg.apply(lambda x: get_edges(x['Phrases']), axis=1)

    # Filters out edges that have less than edge_thresh overlaps
    edge_filtered = {}
    for phrase, phrase_counts in edge_dict.items():
        edge_filtered[phrase] = {}
        for inner_phrase, count in phrase_counts.items():
            if count < edge_thresh:
                continue
            edge_filtered[phrase][inner_phrase] = count

    # Outputs to EdgeData.csv
    outpath_edge = outfolder + '/EdgeData.csv'
    with open(outpath_edge, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Source', 'Target', ' Weight'])
        for phrase, phrase_counts in edge_filtered.items():
            for inner_phrase, count in phrase_counts.items():
                writer.writerow([phrase, inner_phrase, count])

    return


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
