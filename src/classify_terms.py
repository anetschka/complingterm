import argparse
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords
import random

from sklearn.dummy import DummyClassifier
import itertools

import argparse
RANDOMSTATE = 112
random.seed=RANDOMSTATE





class ClassifierExample:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ?".split()
        return sl

    def __init__(self, term,label,freq):
        self.term = wordpunct_tokenize(term.lower())
        self.label = label
        self.freq = freq

    def a_bow(self):
        D = {}
        D["a_headword"] = self.term[-1]
        if len(self.term) > 1:
            for w in self.term[:-1]:
                D["a_bow_"+w]=1
        return D

    def b_len(self):
        D = {}
        D["b_len"] = len(self.term)
        return D

    def featurize(self,variant):
        D = {}
        if "a" in variant:
            D.update(self.a_bow())
        if "b" in variant:
            D.update(self.b_len())
        return D


def crossval(features, labels,variant):
    maxent = LogisticRegression(penalty='l2')
    dummyclass = DummyClassifier("most_frequent")
    scores = defaultdict(list)

    preds = []
    dummypreds = []
    shuffled_gold = []

    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=True):
        # print(TestIndices)
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i = labels[TestIndices]

        shuffled_gold.extend(Testy_i)

        dummyclass.fit(TrainX_i, Trainy_i)
        maxent.fit(TrainX_i, Trainy_i)

        ypred_i = maxent.predict(TestX_i)
        ydummypred_i = dummyclass.predict(TestX_i)
        dummypreds.extend(ydummypred_i)
        acc = accuracy_score(y_true=Testy_i,y_pred=ypred_i)
        f1 = f1_score(y_true=Testy_i,y_pred=ypred_i)
        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        scores["Recall"].append(acc)
        preds.extend(ypred_i)

    print("%s %.3f %.3f" % (variant, np.array(scores["Accuracy"]).mean(), np.array(scores["F1"]).mean()))
    print(classification_report(y_pred=preds,y_true=shuffled_gold))
    print(Counter(preds).most_common())
    print(Counter(labels).most_common())
    print(accuracy_score(y_true=shuffled_gold,y_pred=preds))
    print(scores)



def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../data/classes_for_training.tsv")
    args = parser.parse_args()
    frame = pd.read_csv(args.input,names=["term","label","freq"],sep="\t")
    examples = []
    for t,l,f in zip(list(frame.term),list(frame.label),list(frame.freq)):
        examples.append(ClassifierExample(term=t,label=l,freq=f))

    variant = 'a'
    featuredicts = [ex.featurize(variant) for ex in examples]
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts)
    print(featuredicts[:3])
    labels = np.array([ex.label for ex in examples])
    print(set(labels))

    crossval(features,labels,variant)



if __name__ == "__main__":
    main()
