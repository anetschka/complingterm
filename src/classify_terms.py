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
from nltk.corpus import wordnet as wn
from sklearn.dummy import DummyClassifier
import itertools
from sklearn.metrics import confusion_matrix
import argparse
RANDOMSTATE = 112
random.seed=RANDOMSTATE




class ClassifierExample:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ?".split()
        return sl

    def __init__(self, term,label,freq):
        self.orig_string = term
        self.term = wordpunct_tokenize(term.lower())
        self.label = label
        self.freq = freq

    def __str__(self):
        return " ".join(self.term)+"\t"+self.label

    def a_bow(self):
        D = {}
        D["a_headword"] = self.term[-1]
        if len(self.term) > 1:
            for w in self.term[:-1]:
                D["a_bow_"+w]=1
        return D

    def b_len_and_caps(self):
        D = {}
        D["b_len"] = len(self.term)
        #D["b_caps"] = float(len([x for x in self.orig_string.replace(" ","") if x.isupper()])) / len(self.orig_string)

        return D

    def c_brown(self,brownframe):
        D = {}
        for w in self.term:
            paths = list(brownframe[brownframe.word == w].path)
            if paths:
                #print(w,":::",paths)
                D["c_"+str(paths[0])]=1
            else:
                pass
        return D

    def d_embeddings(self,embeddings):
        D = {}
        A = []
        for w in self.term:
            if w in embeddings.keys():
                A.append(embeddings[w])
        A = np.array(A)
        if A.shape[0] > 1:
            sumvector = A.sum(axis=0) / len(A)
            for i,v in enumerate(sumvector):
                D["d_"+str(i)]=v
        return D
    def e_wordnet(self):
        D = {}
        w = self.term[-1]
        i = 0
        if wn.morphy(w):
            headlemma = wn.morphy(w)
        else:
            headlemma = w
        senses = wn.synsets(headlemma, 'n')
        D["f_numsenses_"+str(i)] = len(senses)
        if len(senses) > 0:
            D["f_headsupersense"+str(i)] = senses[0].lexname()
        else:
            D["f_headsupersense"+str(i)] = "oov"
        return D


    def featurize(self,variant,brownframe,E):
        D = {}
        if "a" in variant:
            D.update(self.a_bow())
        if "b" in variant:
            D.update(self.b_len_and_caps())
        if "c" in variant:
            D.update(self.c_brown(brownframe))
        if "d" in variant:
            D.update(self.d_embeddings(E))
        if "e" in variant:
            D.update(self.e_wordnet())
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
        scores["Accuracy_dummy"].append(accuracy_score(y_true=Testy_i,y_pred=ydummypred_i))
        scores["F1_dummy"].append(f1_score(y_true=Testy_i,y_pred=ydummypred_i))
        preds.extend(ypred_i)

    print("summary %s %.3f %.3f %.3f %.3f" % (variant, np.array(scores["Accuracy"]).mean(), np.array(scores["F1"]).mean(),np.array(scores["Accuracy_dummy"]).mean(), np.array(scores["F1_dummy"]).mean()))
    print(classification_report(y_pred=preds,y_true=shuffled_gold))

    labels_to_print = sorted(set(shuffled_gold))
    CM=confusion_matrix(y_pred=preds,y_true=shuffled_gold,labels=labels_to_print)
    print(sorted(set(shuffled_gold)))
    for l,r in zip(labels_to_print,CM):
        print(l,"\t".join([str(x) for x in r]))
    scores = None
    #print(Counter(preds).most_common())
    #print(Counter(labels).most_common())
    #print(accuracy_score(y_true=shuffled_gold,y_pred=preds))
    #print(scores)


def read_embeddings(infile):
    E = {}
    fin = open(infile)
    header = fin.readline()
    for line in fin.readlines():
        line = line.strip().split(" ")
        E[line[0]] = np.array([float(x) for x in line[1:]])
    return E


def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    #parser.add_argument('--input', default="../data/classes_for_training_23nov.tsv")
    parser.add_argument('--input', default="../data/classes_for_training_coarse.tsv")
    parser.add_argument('--brown', default="../data/paths_brown1000_acl.tsv")
    parser.add_argument('--embeddings', default="../data/embedmodel.txt")

    args = parser.parse_args()

    frame = pd.read_csv(args.input,names=["term","label"],sep="\t")
    brownframe = pd.read_csv(args.brown,names=["path","word","freq"],sep="\t")
    E = read_embeddings(args.embeddings)

    letter_ids = "abcde"
    variants = []
    for k in range(1,6):
        variants.extend(["".join(x) for x in itertools.combinations(letter_ids, k)])

    for variant in variants:
        examples = []
        for t,l in zip(list(frame.term),list(frame.label)):
            ex = ClassifierExample(term=t,label=l,freq=1)
            examples.append(ex)
        print("len examples", len(examples))
        featuredicts = [ex.featurize(variant,brownframe,E) for ex in examples]
        vec = DictVectorizer()
        features = vec.fit_transform(featuredicts)
        labels = np.array([ex.label for ex in examples])
        crossval(features,labels,variant)




if __name__ == "__main__":
    main()
