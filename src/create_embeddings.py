import argparse
from gensim.models import Word2Vec
def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('input')


    args = parser.parse_args()
    sentences  = []
    for line in open(args.input).readlines():
        line = line.strip().split()
        sentences.append(line)

    model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    model.save_word2vec_format("embedmodel.txt")

if __name__ == "__main__":
    main()
