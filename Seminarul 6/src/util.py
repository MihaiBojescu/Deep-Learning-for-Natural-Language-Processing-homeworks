from os import makedirs
from numpy import min, max, ndarray
from os import listdir, makedirs
from random import sample
from re import match
from numpy import ndarray
from nltk.corpus import stopwords


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
    """
    Dimensions:
    - X
    - Y
    - Z
    """
    return sample(range(0, vectorized_words[0].shape[0]), k=k)


def scale_between_0_and_1(vector: ndarray):
    return (vector - min(vector)) / (max(vector) - min(vector))
