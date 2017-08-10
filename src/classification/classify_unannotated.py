import argparse
from classify_terms import ClassifierExample, read_embeddings
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords
import random
from nltk.corpus import wordnet as wn
import itertools
import pandas as pd



def train_and_predics(featuredicts,labels,trainsize):
    vec = DictVectorizer()
    y_train = labels[:trainsize]
    X_train = vec.fit_transform(featuredicts[:trainsize])
    X_test = vec.transform(featuredicts[trainsize:])
    maxent = LogisticRegression(penalty='l2')
    maxent.fit(X_train,y_train)
    predictions = []
    #header = "\t".join(["prediction"]+[str(c) for c in maxent.classes_])
    #predictions.append(header)
    for list,label in zip(maxent.predict_proba(X_test),maxent.predict(X_test)):
        line="\t".join([label]+["{0:.2f}".format(k) for k in list])
        predictions.append(line)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--training', default="../data/classes_for_training_coarse.tsv")
    parser.add_argument('--filetotag', default="../data/non_annotated_terms.txt")

    parser.add_argument('--brown', default="../data/paths_brown1000_acl.tsv")
    parser.add_argument('--embeddings', default="../data/embedmodel.txt")
    parser.add_argument('--variant', default="acde")



    args = parser.parse_args()

    frame = pd.read_csv(args.training,names=["term","label"],sep="\t")
    brownframe = pd.read_csv(args.brown,names=["path","word","freq"],sep="\t")
    E = read_embeddings(args.embeddings)
    variant = args.variant


    examples = []
    for t,l in zip(list(frame.term),list(frame.label)):
        ex = ClassifierExample(term=t,label=l,freq=1)
        examples.append(ex)

    trainsize = len(examples)

    newterms = []
    for line in open(args.filetotag).readlines():
        t = line.strip()
        newterms.append(t)
        ex = ClassifierExample(term=t,label="blank",freq=1)
        examples.append(ex)

    featuredicts = [ex.featurize(variant,brownframe,E) for ex in examples]
    labels = np.array([ex.label for ex in examples])
    predlines=train_and_predics(featuredicts, labels, trainsize)
    for t,l in zip(newterms,predlines):
        print(t+"\t"+l)


if __name__ == "__main__":
    main()
