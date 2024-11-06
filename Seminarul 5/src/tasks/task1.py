#!/usr/bin/env python

from os import listdir, makedirs
from re import match
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def task1():
    docs = read()
    preprocessed_docs = preprocess(docs)

    save(preprocessed_docs)

    bow_matrix, terms = vectorize(preprocessed_docs)

    return bow_matrix, terms


def read():
    result: list[str] = []

    for raw_file in listdir("./data/raw"):
        with open(f"./data/raw/{raw_file}", "r", encoding="utf-8") as file:
            result.append(file.read())

    return result


def preprocess(docs: list[str]):
    lemmatizer = WordNetLemmatizer()
    accumulator: list[str] = []
    result: list[str] = []

    for doc in docs:
        words = doc.lower().split(" ")
        accumulator.clear()

        for word in words:
            matches = match(r"(\w+)", word)

            if matches is None or len(matches.groups()) != 1:
                continue

            if matches[0] in stopwords.words("english"):
                continue

            lemmatized_word = lemmatizer.lemmatize(matches[0])
            accumulator.append(lemmatized_word)

        result.append(" ".join(accumulator))

    return result


def save(preprocessed_docs: list[str]):
    makedirs("./data/lemmatized", exist_ok=True)

    for preprocessed_doc, raw_file in zip(preprocessed_docs, listdir("./data/raw")):
        with open(f"./data/lemmatized/{raw_file}", "w", encoding="utf-8") as file:
            file.write(preprocessed_doc)


def vectorize(preprocessed_docs: list[str]):
    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(preprocessed_docs)
    terms = bow_vectorizer.get_feature_names_out()
    return bow_matrix, terms


if __name__ == "__main__":
    task1()
