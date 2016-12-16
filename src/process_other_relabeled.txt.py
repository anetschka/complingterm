import argparse
from gensim.models import Word2Vec

def filter(line):

    traits = [x for x in line.strip().split("\t") if len(x) > 0]
    if len(traits) > 4:
        label = "/".join(sorted([traits[-1],traits[-2]]))
        print("\t".join([traits[0],label,"_"]))


def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input',default="../data/other_relabel.txt")


    args = parser.parse_args()
    sentences  = []
    for line in open(args.input).readlines():
          filter(line)

if __name__ == "__main__":
    main()
