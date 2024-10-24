from lib import LaPlaceSmoothing, NGrams, NGramResults


def main():
    corpus = read_corpus()
    sentence = read_input()

    smoothing_function = LaPlaceSmoothing()
    trigrams = NGrams(3, smoothing_function)

    corpus_result = trigrams.generate_ngrams(corpus)
    sentence_result = trigrams.generate_ngrams([sentence])

    probability = calculate_probability(corpus_result, sentence_result)

    print(f"Probability: {probability}")


def read_corpus() -> list[str]:
    with open("./data/corpus.txt", mode="r", encoding="utf-8") as file:
        return file.readlines()


def read_input() -> str:
    return input("Sentence: ")


def calculate_probability(
    corpus_result: NGramResults, sentence_result: NGramResults
) -> float:
    probability = 1.0

    for ngram in sentence_result.ngrams:

        if ngram in corpus_result.smoothed_frequencies:
            probability *= corpus_result.smoothed_frequencies[ngram]
        else:
            probability *= 1 / (
                corpus_result.vocabulary_size + len(corpus_result.smoothed_frequencies)
            )

    return probability


if __name__ == "__main__":
    main()
