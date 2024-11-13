from gensim.models import KeyedVectors
from numpy import dot, ndarray, linalg
from util import vectorize


def task2():
    word_pairs = [
        ("car", "automobile"),
        ("wrench", "screwdriver"),
        ("disk", "vinyl"),
        ("happiness", "sadness"),
        ("clothes", "apartments"),
        ("country", "rock"),
        ("laptop", "door"),
    ]

    for model_name in [
        "glove-wiki-gigaword-50",
        "word2vec-google-news-300",
    ]:
        model = KeyedVectors.load(f"./models/{model_name}.model", mmap="r")
        vectorized_word_pairs: list[tuple[ndarray, ndarray]] = [
            tuple(vectorize(model, list(word_pair))) for word_pair in word_pairs
        ]

        cosine_similarity(model_name, word_pairs, vectorized_word_pairs)


def cosine_similarity(
    model_name: str,
    word_pairs: list[tuple[str, str]],
    vectorized_word_pairs: list[tuple[ndarray, ndarray]],
):
    for word_pair, vectorized_word_pair in zip(word_pairs, vectorized_word_pairs):
        dot_product = dot(vectorized_word_pair[0], vectorized_word_pair[1])
        magnitude_1 = linalg.norm(vectorized_word_pair[0])
        magnitude_2 = linalg.norm(vectorized_word_pair[1])

        cosine_similarity = dot_product / (magnitude_1 * magnitude_2)

        print(
            f"Cosine Similarity for word pair ({", ".join(word_pair)}) and model \"{model_name}\": {cosine_similarity:.5f}"
        )
