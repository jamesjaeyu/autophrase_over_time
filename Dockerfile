# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

FROM $BASE_CONTAINER

LABEL maintainer="James Yu <jjy002@ucsd.edu>"

# 2) change to root to install packages
USER root

#RUN apt-get update && apt-get -y install openjdk-8-jdk chromium-chromedriver

# 3) install packages using notebook user
USER jovyan

#RUN conda install --yes scikit-learn

RUN pip install --no-cache-dir pandas matplotlib sklearn Levenshtein gensim nltk

# Override command to disable running jupyter notebook at launch
#CMD ["/bin/bash"]
