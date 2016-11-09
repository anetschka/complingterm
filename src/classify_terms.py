import argparse
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.preprocessing import Normalizer
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords

from sklearn.dummy import DummyClassifier
import itertools

import argparse




def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../data/classes_for_training.tsv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input,names=["term","class","freq"],sep="\t")
    print(list(frame.term)[:30])


if __name__ == "__main__":
    main()
