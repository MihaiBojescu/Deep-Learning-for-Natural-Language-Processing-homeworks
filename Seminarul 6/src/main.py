#!/usr/bin/env python

from os import listdir, makedirs
from random import sample
from re import match
from gensim.models import KeyedVectors
from numpy import array, min, max, ndarray, stack
from matplotlib import pyplot as plt
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords


def main():
    make_dirs()

    dataset = read()
    words = pick_words(dataset, 20)

    print(words)

    for model in [
        KeyedVectors.load("./models/glove-wiki-gigaword-50.model", mmap="r"),
        KeyedVectors.load("./models/word2vec-google-news-300.model", mmap="r"),
    ]:
        vectorized_words = vectorize(model, words)
        dimensions = pick_dimensions(vectorized_words)

        plot_random_dimensions(words, vectorized_words, dimensions)


def make_dirs():
    makedirs("./outputs", exist_ok=True)


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

        if (
            matches is None
            or len(matches.groups()) != 1
            or matches[0] in stopwords.words("english")
        ):
            continue

        words.append(matches[0])

    picked_words = sample(words, k=k)
    return picked_words


def vectorize(model, words: list[str]) -> list[ndarray]:
    result: list[ndarray] = []

    for word in words:
        try:
            result.append(model[word])
        except:
            print(f"Unvectorizable word: {word}")

    return result


def pick_dimensions(vectorized_words, k=3):
    return sample(range(0, vectorized_words[0].shape[0]), k=k)


if __name__ == "__main__":
    main()
