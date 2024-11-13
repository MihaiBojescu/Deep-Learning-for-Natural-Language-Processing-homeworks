#!/usr/bin/env python

from gensim.models import KeyedVectors
import nltk

nltk.download("stopwords")

from util import make_dirs, read, pick_words, vectorize, pick_dimensions
from task1 import plot_random_dimensions, plot_pca, plot_tsne


def main():
    make_dirs()

    dataset = read()
    words = pick_words(dataset, 20)

    print(f"The words are: {", ".join(words)}.")

    for model_name in [
        "glove-wiki-gigaword-50",
        "word2vec-google-news-300",
    ]:
        model = KeyedVectors.load(f"./models/{model_name}.model", mmap="r")
        vectorized_words = vectorize(model, words)
        dimensions = pick_dimensions(vectorized_words)

        plot_random_dimensions(model_name, words, vectorized_words, dimensions)
        plot_pca(model_name, words, vectorized_words)
        plot_tsne(model_name, words, vectorized_words)


if __name__ == "__main__":
    main()
