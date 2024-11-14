#!/usr/bin/env python

import nltk

nltk.download("stopwords")

from task1 import task1
from task2 import task2


def main():
    task1()
    task2()


if __name__ == "__main__":
    main()
