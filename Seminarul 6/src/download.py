#!/usr/bin/env python

from os import makedirs
from gensim import downloader


def main():
    makedirs("./models/", exist_ok=True)

    for name in [
        "word2vec-google-news-300",
        "glove-wiki-gigaword-50",
    ]:
        model = downloader.load(name)
        model.save(f"./models/{name}.model")


if __name__ == "__main__":
    main()
