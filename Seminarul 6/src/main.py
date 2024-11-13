#!/usr/bin/env python

from os import listdir
from random import choices
from re import match
from gensim import downloader
from numpy import ndarray


def main():
    dataset = read()
    words = pick_words(dataset, 20)

    print(words)

    for model in [
        downloader.load("glove-wiki-gigaword-50"),
        downloader.load("word2vec-google-news-300"),
    ]:
        vectorized_words = vectorize(model, words)
        dimensions = pick_dimensions(vectorized_words)

        print(vectorized_words)
        print(dimensions)


def read():
    dataset: str = ""

    for filename in listdir("./data"):
        with open(f"./data/{filename}", "r", encoding="utf-8") as file:
            dataset += file.read()

    return dataset


def pick_words(dataset: str, k=20):
    words: list[str] = []

    for word in dataset.lower().split(" "):
        matches = match(r"(\w+)", word)

        if matches is None or len(matches.groups()) != 1:
            continue

        words.append(matches[0])

    picked_words = choices(words, k=k)
    return picked_words


def vectorize(model, words: list[str]) -> list[ndarray]:
    result: list[ndarray] = []

    for word in words:
        result.append(model[word])

    return result


def pick_dimensions(vectorized_words, k=3):
    return choices(range(0, vectorized_words[0].shape[0]), k=k)


if __name__ == "__main__":
    main()
