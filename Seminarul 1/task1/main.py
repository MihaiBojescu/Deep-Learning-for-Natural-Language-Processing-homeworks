#!/usr/bin/env python3

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from dataclasses import dataclass


@dataclass
class WordnetResult:
    word: str
    synonyms: set[str]
    antonyms: set[str]
    hypernyms: set[str]
    meronyms: set[str]

    def __str__(self):
        return f"""
Results for word "{self.word}":
    Synonyms: {", ".join(self.synonyms)}
    Antonyms: {", ".join(self.antonyms)}
    Hypernyms: {", ".join(self.hypernyms)}
    Meronyms: {", ".join(self.meronyms)}
        """


def main():
    # nltk.download()
    # nltk.download("punkt_tab")
    # nltk.download("wordnet")

    sentence = read()
    results = analyser(sentence)

    for result in results:
        print(str(result))


def read():
    sentence = input("Sentence for analysis: ")
    sentence = sentence.strip()

    return sentence


def analyser(sentence: str):
    tokens = word_tokenize(sentence)
    results: list[WordnetResult] = []

    for token in tokens:
        synonyms: list[str] = []
        antonyms: list[str] = []
        hypernyms: list[str] = []
        meronyms: list[str] = []

        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                synonyms.append(lemma.name())
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())

            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    hypernyms.append(lemma.name())

            for part_meronym in synset.part_meronyms():
                for lemma in part_meronym.lemmas():
                    meronyms.append(lemma.name())

        results.append(
            WordnetResult(
                word=token,
                synonyms=set(synonyms),
                antonyms=set(antonyms),
                hypernyms=set(hypernyms),
                meronyms=set(meronyms),
            )
        )

    return results


if __name__ == "__main__":
    main()
