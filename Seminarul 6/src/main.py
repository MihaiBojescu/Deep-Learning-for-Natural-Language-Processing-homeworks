#!/usr/bin/env python

from gensim.models import KeyedVectors
import nltk

nltk.download("stopwords")

from task1 import task1


def main():
    task1()


if __name__ == "__main__":
    main()
