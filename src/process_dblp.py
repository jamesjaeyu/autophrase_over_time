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
    count = 0
    data = []
    first_line = True
    tags = ['"title"', '"year"', '"keywords"', '"abstract"', '"fos"']
    invalid_tags = ['"venue" :', '"_id" :', '"type" :', '"raw" :', '"raw_zh" :',
                    '"n_citation" :', '"page_start" :', '"page_end" :',
                    '"lang" :', '"volume" :', '"issue" :', '"issn" :',
                    '"isbn" :', '"doi" :', '"pdf" :', '"url" :', '"references" :']
    valid_tag = True
    num_papers = 0

    with open(inpath, 'r', encoding='utf-8') as f:
        content = '{'
        # Keeps track of the start of processing a block (individual paper)
        start_ind = False
        for line in f:
            # TODO: need to fix issue with paper #3408

            # Indicates start of block
            if line == '{ \n':
                start_ind = True

            # Indicates end of block
            if line == '},\n':
                # Doesn't process the first block (only contains info about structure)
                if first_line:
                    first_line = False
                    content = '{'
                    continue
                # Cleans up formatting of block
                content += '}\n'
                content = content.replace(' :', ':').replace(',', ', ')
                # Prevents issue with additional comma at end of dict
                if content[-4:-2] == ', ':
                    content = content[:-4] + '}\n'
                # Fix for issue with double beginning brackets
                if content[:2] == '{{':
                    content = content[1:]

                # for testing purposes
                num_papers += 1
                print(str(num_papers) + ' papers processed')
                try:
                    json.loads(content)
                except:
                    print(content)
                    print('--------------- ISSUE HAS OCCURRED --------------')
                    break
                #data.append(json.loads(content))
                print('----------------------')

                start_ind = False
                content = '{'

            if start_ind:
                # Cleans up line to prevent issues with json.loads()
                line = line.strip()
                print(line)
                if any([tag in line for tag in tags]):
                    #line = line
                    valid_tag = True
                #elif any([tag in invalid_tags for tag in tags]):
                # TODO: fix issue where '"text : text' causes this case to be True
                #       as it causes scanning to stop too early
                elif ' : ' in line:
                    valid_tag = False
                if not valid_tag:
                    continue

                if 'NumberInt' in line:
                    line = line.replace('NumberInt(', '')
                    line = line.strip()[:-2] + ','
                if line[-3:] == ': ,':
                    line = line.replace(': ,', ': null')
                if line == '"type" : 1,':
                    continue
                #print(line)
                content += line
    return data


# TODO
# 1. Obtain the number of papers per year (value counts)
# 2. Output the title + abstracts to aggregated .txt files by year
# 3. Output unique keywords to aggregated .txt files by year
#    (Could also try including duplicates so we can see the most common keywords by year)

