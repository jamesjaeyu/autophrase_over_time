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

    Only care about: title, year, keywords, abstract
    When we encounter one of these tags, we want to keep processing lines until we reach an invalid tag

    >>> process_json('../data/dblpv13.json')
    """
    #count = 0
    data = []
    first_line = True
    tags = ['title', 'year', 'keywords', 'abstract', 'fos']
    invalid_tags = ['"venue" :', '"_id" :', '"type" :', '"raw" :', '"raw_zh" :',
                    '"ncitation" :', '"page_start" :', '"page_end" :',
                    '"lang" :', '"volume" :', '"issue" :', '"issn" :',
                    '"isbn" :', '"doi" :']
    valid_tag = True

    with open(inpath, 'r', encoding='utf-8') as f:
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
                try:
                    json.loads(content)
                except:
                    print('--------------- ISSUE HAS OCCURRED --------------')
                    break
                #data.append(json.loads(content))
                print('----------------------')
                start_ind = False
                content = ''

            if start_ind:
                # Cleans up line to prevent issues with json.loads()
                line = line.strip()
                if any([tag in line for tag in tags]):
                    valid_tag = True
                else:
                    valid_tag = False
                if 'NumberInt' in line:
                    line = line.replace('NumberInt(', '')
                    line = line.strip()[:-2] + ','
                if line[-3:] == ': ,':
                    line = line.replace(': ,', ': null')
                if line == '"type" : 1,':
                    continue
                print(line)
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

