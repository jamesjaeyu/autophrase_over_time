"""
DSC180B
Q2 Project
Backup for any unused functions
"""

# NOTE: model_generation.py functions for processing phrasal segmentation results
# Using pandas is far more efficient than reading through the .txt files line by line
def process_seg(infolder):
    """
    Processes segmentation.txt files in all subfolders of infolder

    Creates csv with columns: Phrases, Year, Frequency
    Each entry of Phrases will contain all phrases within a single paper

    TODO: Issue with low-quality phrases being included in output. This can be
          resolved during processing or afterwards.

    >>> process_seg('../results/dblp-v10-grouped')
    """
    start = time.time()
    # Obtains filepaths for all segmentation.txt files
    filepaths = []
    for i in range(1950, 2019, 5):
        if i == 1950 or i == 1955:
            subfolder = '1950-1959'
        elif i == 2015:
            subfolder = '2015-2017'
        else:
            subfolder = str(i) + '-' + str(i+4)
        filepaths.append(infolder + '/' + subfolder + '/segmentation.txt')

    df = pd.DataFrame(columns=['Phrases', 'Year'])
    for fp in filepaths:
        year = fp.split('/')[3]
        file = open(fp)
        # Each line in the file represents a single paper's title + abstract
        for line in file:
            phrases = []
            line = line.lower()
            # Adds marked phrases to the data list until we have no more phrases
            while line.find('<phrase>') != -1:
                start_idx = line.find('<phrase>')
                end_idx = line.find('</phrase>')
                phrase = line[start_idx+8:end_idx]
                phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
                phrases.append(phrase)
                line = line[end_idx+9:]
            # Phrases are separated by commas
            phrases = ','.join(phrases)
            df.loc[len(df.index)] = [phrases, year]
    outpath = infolder + '/dblp-v10-grouped-seg.csv'
    df.to_csv(outpath)
    end = time.time()
    return end - start

def process_seg_alt(infolder):
    """
    Alternate approach to processing segmentation.txt files
    Outputs separate csvs for each year range, rather than all in a single csv
    (Takes around 7.5 hours to run)

    >>> process_seg_alt('../results/dblp-v10-grouped')
    """
    start = time.time()
    # Obtains filepaths for all segmentation.txt files
    filepaths = []
    for i in range(1950, 2019, 5):
        if i == 1950 or i == 1955:
            subfolder = '1950-1959'
        elif i == 2015:
            subfolder = '2015-2017'
        else:
            subfolder = str(i) + '-' + str(i+4)
        filepaths.append(infolder + '/' + subfolder + '/segmentation.txt')

    for fp in filepaths:
        df = pd.DataFrame(columns=['Phrases', 'Year'])
        file = open(fp)
        year = fp.split('/')[3]
        # Each line in the file represents a single paper's title + abstract
        for line in file:
            phrases = []
            line = line.lower()
            # Adds marked phrases to the data list until we have no more phrases
            while line.find('<phrase>') != -1:
                start_idx = line.find('<phrase>')
                end_idx = line.find('</phrase>')
                phrase = line[start_idx+8:end_idx]
                phrase = re.sub(r'[^A-Za-z0-9- ]+', '', phrase)
                phrases.append(phrase)
                line = line[end_idx+9:]
            # Phrases are separated by commas
            phrases = ','.join(phrases)
            df.loc[len(df.index)] = [phrases, year]
        outpath = infolder + '/segmentation/' + year + '_segmented.csv'
        df.to_csv(outpath)
    end = time.time()
    return end - start
