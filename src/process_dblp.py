"""
DSC180B
Q2 Project
"""
import pandas as pd
from random import sample
import sys
import json

def process_json(inpath):
    """
    Process the inpath of DBLP v13 dataset
    Outputs a list of dictionaries (each dictionary is a paper)

    >>> process_json('../data/dblpv13.json')
    """
    #count = 0
    data = []
    first_line = True

    with open(inpath) as f:
        content = ''
        # Keeps track of the start of processing a block (individual paper)
        start_ind = False
        for line in f:
            # Indicates start of block
            if line == '{ \n':
                start_ind = True
            # Indicates end of block
            if line == '},\n':
                # Doesn't process the first block (only contains info about structure)
                if first_line:
                    first_line = False
                    content = ''
                    continue
                # Cleans up formatting of block
                content += '}\n'
                content = content.replace(' :', ':').replace(',', ', ')
                data.append(json.loads(content))
                start_ind = False
                content = ''

            if start_ind:
                # Cleans up line to prevent issues with json.loads()
                line = line.strip()
                if 'NumberInt' in line:
                    line = line.replace('NumberInt(', '')
                    line = line.strip()[:-2] + ','
                if line[-3:] == ': ,':
                    line = line.replace(': ,', ': null')
                content += line

            #count += 1
            #if count > 200:
            #    break
    return data


# TODO
# 1. Obtain the number of papers per year (value counts)
# 2. Output the title + abstracts to aggregated .txt files by year
# 3. Output unique keywords to aggregated .txt files by year
#    (Could also try including duplicates so we can see the most common keywords by year)

