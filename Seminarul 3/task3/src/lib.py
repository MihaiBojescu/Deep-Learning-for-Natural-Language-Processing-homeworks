from collections import defaultdict
from dataclasses import dataclass
from re import sub
from typing import Callable


class LaPlaceSmoothing:
    def __call__(
        self, ngram_frequencies: defaultdict[tuple[str, ...], int], vocabulary_size: int
    ) -> defaultdict[tuple[str, ...], float]:
        smoothed_ngram_frequencies: defaultdict[tuple[str, ...], float] = defaultdict(
            float
        )

        for ngram in ngram_frequencies:
            smoothed_ngram_frequencies[ngram] = (ngram_frequencies[ngram] + 1) / (
                vocabulary_size + len(ngram_frequencies)
            )

        return smoothed_ngram_frequencies


@dataclass
class NGramResults:
    ngrams: list[tuple[str, ...]]
    vocabulary_size: int
    frequencies: defaultdict[tuple[str, ...], float]
    smoothed_frequencies: defaultdict[tuple[str, ...], float]


class NGrams:
    __n: int
    __smoothing_function: Callable[[defaultdict[tuple[str, ...], int], int], float]

    def __init__(
        self,
        n: int,
        smoothing_function: Callable[[defaultdict[tuple[str, ...], int], int], float],
    ) -> None:
        self.__n = n
        self.__smoothing_function = smoothing_function

    def generate_ngrams(self, corpus: list[str]) -> NGramResults:
        ngrams = self.__generate_corpus_ngrams(corpus)
        ngram_frequencies = self.__calculate_ngram_frequencies(corpus)
        vocabulary_size = self.__calculate_vocabulary_size(corpus)
        smoothed_ngram_frequencies = self.__smoothing_function(
            ngram_frequencies, vocabulary_size
        )

        return NGramResults(
            ngrams=ngrams,
            vocabulary_size=vocabulary_size,
            frequencies=ngram_frequencies,
            smoothed_frequencies=smoothed_ngram_frequencies,
        )

    def __calculate_vocabulary_size(self, corpus: list[str]) -> int:
        return len(
            set(
                word
                for sentence in corpus
                for word in self.__preprocess(sentence).split()
            )
        )

    def __generate_corpus_ngrams(self, corpus: list[str]) -> list[tuple[str, ...]]:
        result: list[tuple[str, ...]] = []

        for sentence in corpus:
            preprocessed_sentence = self.__preprocess(sentence)
            ngrams = self.__generate_ngrams(preprocessed_sentence)
            result.extend(ngrams)

        return list(set(result))

    def __calculate_ngram_frequencies(
        self, corpus: list[str]
    ) -> defaultdict[tuple[str, ...], int]:
        ngram_frequencies: defaultdict[tuple[str, ...], int] = defaultdict(int)

        for sentence in corpus:
            preprocessed_sentence = self.__preprocess(sentence)
            ngrams = self.__generate_ngrams(preprocessed_sentence)

            for ngram in ngrams:
                ngram_frequencies[ngram] += 1

        return ngram_frequencies

    def __preprocess(self, text: str) -> str:
        text = text.lower()
        text = sub(r"\W+", " ", text)
        text = text.strip()
        return text

    def __generate_ngrams(self, text: str) -> list[tuple[str, ...]]:
        words = text.split()
        ngrams: list[tuple[str, ...]] = []

        for i in range(0, len(words) - self.__n + 1):
            ngram: tuple[str, ...] = tuple(words[j] for j in range(i, i + self.__n))
            ngrams.append(ngram)

        return ngrams
