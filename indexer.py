import collections
import csv
import gzip
import json
import os
import sys
import math


# BM25 and QL models


class Posting:
    """A Posting is a pair of a document ID and a list of positions.

    Attributes:
      doc_id: a string representing the document ID
      positions: a list of integers representing the positions of the term in the
        document
    """

    def __init__(self, doc_id, positions, termCount, docLen, score, N):
        self.doc_id = doc_id
        self.positions = positions
        self.termCount = termCount
        self.docLen = docLen
        self.score = score
        self.N = N

    def __repr__(self):
        return f"Posting({self.doc_id}, {self.positions},{self.docLen}, {self.score})"

    def __eq__(self, other):
        return self.doc_id == other.doc_id and self.positions == other.positions

    def __hash__(self):
        return hash((self.doc_id, tuple(self.positions)))


def read_data(filename):

    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            data = json.load(f)["corpus"]
        return data
    elif filename.endswith('.tsv'):
        data = []
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                data.append(line.split('\t'))
        return data

    else:
        with open(filename, 'r') as f:
            data = json.load(f)["corpus"]
        return data


def build_index(inputFile):
    """Builds an index from the given input file.

    Args:
      inputFile: a string representing the path to the input file

    Returns:
      A dictionary mapping terms to a list of postings.
    """

    corpus = read_data(inputFile)

    index = collections.defaultdict(list)
    collectionLength = 0
    cqi = {}
    for doc in corpus:

        doc_index = collections.defaultdict(list)
        for i, term in enumerate(doc["text"].split()):
            doc_index[term].append(i)
            collectionLength += 1

        for term, positions in doc_index.items():
            index[term].append(Posting(doc["sceneId"], positions, len(
                positions), len(doc["text"].split()), 0, len(corpus)))
            if term not in cqi:
                cqi[term] = len(positions)
            else:
                cqi[term] += len(positions)

        # get the number of times the term appears in the entire corpus

    return index, corpus, collectionLength, cqi


def bm25_ranking(index, queryTerms, corpus, collectionLength):
    avgdl = collectionLength / len(corpus)
    N = len(corpus)
    k1 = 1.8
    k2 = 5
    b = 0.75

    uniqueQueryTerms = []
    top100 = []

    for term in queryTerms:
        if term not in uniqueQueryTerms:
            uniqueQueryTerms.append(term)

    for docs in corpus:
        score = 0
        for term in uniqueQueryTerms:
            ni = len(index[term])  # ni = number of documents containing term i
            qfi = queryTerms[term]
            for posting in index[term]:
                if posting.doc_id == docs["sceneId"]:
                    fi = posting.termCount
                    dl = posting.docLen

                    K = k1 * ((1 - b) + b * (dl / avgdl))
                    first = math.log((N - ni + 0.5) / (ni + 0.5), math.e)
                    second = ((k1 + 1) * fi) / (K + fi)
                    third = ((k2 + 1) * qfi) / (k2 + qfi)
                    score += first * second * third
                    posting.score = score
                    top100.append(posting)

    top100.sort(key=lambda x: x.score, reverse=True)
    top100 = top100[:100]

    return top100


def ql_ranking(index, terms, corpus, collectionLength, cqi):

    # fqi = posting.termCount
    #cardD = posting.docLen
    #cardC = collectionLength
    # cqi = # of times query term appears in corpus

    cardC = collectionLength
    mue = 300
    top100 = []

    for docs in corpus:
        score = 0
        for term in terms:
            for posting in index[term]:
                if posting.doc_id == docs["sceneId"]:
                    fqi = posting.termCount
                    cardD = posting.docLen

                    # print("cqi: " + str(cqi[term]) + " fqi: " + str(fqi) +
                    #       " cardD: " + str(cardD) + " cardC: " + str(cardC) + " term: " + term + " doc: " + docs["sceneId"])
                    score = math.log((fqi + mue * (cqi[term] / cardC)) /
                                     (cardD + mue), math.e)
                    posting.score += score
                    top100.append(posting)
                else:
                    fqi = 0
                    cardD = posting.docLen
                    print("cqi: " + str(cqi[term]) + " fqi: " + str(fqi) +
                          " cardD: " + str(cardD) + " cardC: " + str(cardC) + " term: " + term + " doc: " + docs["sceneId"])
                    score = math.log((fqi + mue * (cqi[term] / cardC)) /
                                     (cardD + mue), math.e)
                    posting.score += score
                    top100.append(posting)

    top100.sort(key=lambda x: x.score, reverse=True)
    top100 = top100[:100]

    return top100


def main(inputFile, queriesFile, outputFolder):
    index, corpus, collectionLength, cqi = build_index(inputFile)

    with open(queriesFile, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            with open(f"{outputFolder}/{row[0]}.txt", "w") as f:
                terms = collections.Counter(row[3:])

                if row[2] == 'bm25':
                    for posting in bm25_ranking(
                            index, terms, corpus, collectionLength):

                        f.writelines(
                            f"{row[0]}\t{row[1]}\t{posting.doc_id}\t{posting.score} \n")

                elif row[2] == 'ql':
                    for posting in ql_ranking(
                            index, terms, corpus, collectionLength, cqi):
                        f.writelines(
                            f"{row[0]}\t{row[1]}\t{posting.doc_id}\t{posting.score} \n")

    # wrtie to file

    # rank top 10 scores with no ties in score


# main("shakespeare-scenes.json.gz", "./trainQueries.tsv", "/results")
# main("testing.json", "./test.tsv", "/results")
# main("shakespeare-scenes.json.gz", "./test.tsv", "/resultsFolder")
# main("shakespeare-scenes.json.gz", "./test2.tsv", "/resultsFolder")
# main("shakespeare-scenes.json.gz", "./test3.tsv", "/resultsFolder")


if __name__ == "__main__":
    # Read arguments from command line, or use sane defaults for IDE.
    argv_len = len(sys.argv)
    inputFile = sys.argv[1] if argv_len >= 2 else "shakespeare-scenes.json.gz"
    queriesFile = sys.argv[2] if argv_len >= 3 else "trainQueries.tsv"
    outputFolder = sys.argv[3] if argv_len >= 4 else "results/"
    if not os.path.isdir(outputFolder):
        os.mkdir(outputFolder)
    main(inputFile, queriesFile, outputFolder)
