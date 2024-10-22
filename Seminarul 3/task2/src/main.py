from pprint import pprint
from lib import LaPlaceSmoothing, NGrams


def main():
    corpus = read_corpus()

    smoothing_function = LaPlaceSmoothing()
    trigrams = NGrams(2, smoothing_function)

    result = trigrams.generate(corpus)

    pprint(result)


def read_corpus() -> list[str]:
    with open("./data/corpus.txt", mode="r", encoding="utf-8") as file:
        return file.readlines()


if __name__ == "__main__":
    main()
