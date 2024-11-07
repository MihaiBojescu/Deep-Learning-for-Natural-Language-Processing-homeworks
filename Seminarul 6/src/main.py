#!/usr/bin/env python

from os import listdir
from random import choices
from gensim import downloader


def main():
    glove_vectors = downloader.load("glove-wiki-gigaword-50")
    word2vec_vectors = downloader.load("word2vec-google-news-300")

    words = pick_words()


def pick_words():
    docs: list[str] = []

    for filename in listdir("./data"):
        with open(f"./data/{filename}", "r", encoding="utf-8") as file:
            docs.extend(file.read())

    words = choices(docs, k=20)
    return words


def vectorize(word: str):
    pass


if __name__ == "__main__":
    main()
