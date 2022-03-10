# Utilizing AutoPhrase on Computer Science papers over time
### DSC 180B Capstone Project

### Group Members: Cameron Brody, Jason Lin, James Yu

### Link to website: https://jamesjaeyu.github.io/autophrase_over_time/

<br />

[Link to DBLP v10 dataset download](https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip)

- Direct download link from [aminer.org/citation](https://www.aminer.org/citation)
- zip file size is 1.7 GB, 4.08 GB when extracted

File & folder descriptions:
- `config` folder: Contains configuration files for run.py

- `data` folder: Directory for full data when running 'all' target on run.py

- `docs` folder: Contains GitHub Pages files for the visual presentation website

- `results` folder: Contains results of EDA figures and AutoPhrase

- `src` folder: Contains .py files for processing datasets, performing EDA, and model generation

- `test` folder: Contains `testdata` and `testresults` for running 'test' target on run.py

- `run.py`: Run file for the project. Targets are 'data', 'eda', 'model', 'analysis'
    - 'all' will run all targets with full dataset
    - 'test' will run 'data', 'model', and 'analysis' with test data. 'eda' will run the same
