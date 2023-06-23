# Platform-Labeler

This repository contains a code example for a project that trains a neural network to label Brazilian mayoral electoral platforms at the sentence level.

The repositiory contains the following items:
  1) doc_labeler.py - the main code. Run this script to train the algorithms and label the sample documents
  2) network_trainer.py - script that contains the functions necessary to train the algorithms
  3) tokenizer.py - a script that contains the functions necessary to parse a raw txt file and generate sentence embeddings
  4) txt - a folder containing sample platforms in plaintext format
  5) complete_labeles.csv - labeled platform-sentences used to train the algorithms

Running doc_labler.py will generate 2 new files: ROC.png, a plot of the receiver operating characteristic curves for each binary classifier, and an output file with the following name format: [date]_[word embedding model].csv

Dependencies: pandas, torch, torchnlp, nltk, sklearn
