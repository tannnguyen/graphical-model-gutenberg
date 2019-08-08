# Graphical Models with Gutenberg Dataset

## Overview

While nowadays machine learning focuses more on deep learning, some natural language processing (NLP) tasks tend to use different models, namely graphical models. The most two basic tasks of NLP, topic modeling and language modeling, use a variation of graphical models. We want to analyze different models to see its effectiveness on each task. The full report is [here](https://github.com/tannnguyen/graphical-model-gutenberg/blob/master/report.pdf).

## Topic Modeling

Topic modeling aims to "summarize" books and documents into smaller sets of topics, which hopefully can explain the context of these documents. For Gutenberg dataset, we hope to see if topic modeling can generalize the main topics that these authors write about. We consider two graphical models, Latent Dirichlet Allocation (LDA), which is considered state of the art, Correlation Explanation (CoRex). We evaluate with numeric results using topic coherence metric and literary results using what literature critics have described about these authors.

## Language Modeling

Language modeling aims to predict the next word of the sentence given its previous context. Similar to topic modeling, we trained on individual authors, allowing the model to learn from those context and test on another text. We will use two graphic models, Restricted BoltzmannMachines and Log Bilinear Language Model. We evaluate these models with perplexity score or the number of words that model considered when choosing the next word. The lower the score, the more confident the model has on its prediction, the better the model is. 

## System

Codes are written to run on MacOS, using Python 3.6. Packages include NLTK 3.4, Pytorch 1.0

## Author

* Tan Nguyen

## Reference

We used the implementation of Log Bilinear Language Models [here](https://github.com/wenjieguan/Log-bilinear-language-models).
