from gensim.models import KeyedVectors
from numpy import dot, ndarray, linalg
from matplotlib import pyplot as plt

from util import vectorize


def task2():
    word_pairs = [
        ("car", "automobile"),
        ("wrench", "screwdriver"),
        ("disk", "vinyl"),
        ("happiness", "sadness"),
        ("clothes", "apartments"),
        ("pasta", "rock"),
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

        similarities = cosine_similarity(vectorized_word_pairs)
        plot(model_name, word_pairs, similarities)


def cosine_similarity(
    vectorized_word_pairs: list[tuple[ndarray, ndarray]],
):
    results: list[float] = []

    for vectorized_word_pair in vectorized_word_pairs:
        dot_product = dot(vectorized_word_pair[0], vectorized_word_pair[1])
        magnitude_1 = linalg.norm(vectorized_word_pair[0])
        magnitude_2 = linalg.norm(vectorized_word_pair[1])

        similarity = dot_product / (magnitude_1 * magnitude_2)

        results.append(similarity)

    return results


def plot(
    model_name: str,
    word_pairs: list[tuple[str, str]],
    similarities: list[float],
):
    for word_pair, similarity in zip(word_pairs, similarities):
        print(
            f'Cosine Similarity for word pair ({", ".join(word_pair)}) and model "{model_name}": {similarity:.5f}'
        )

    xticks = ["\nand\n".join(word_pair) for word_pair in word_pairs]

    plt.tight_layout()

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.35)
    ax = fig.add_subplot()

    ax.bar(xticks, similarities)
    ax.set_yticks([x / 10 for x in range(0, 11)])
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_xlabel("Word pair")
    ax.set_ylabel("Similarity [0, 1]")
    ax.set_title(f"Similarity between word pairs for model \"{model_name}\"")


    fig.savefig(
        f"./outputs/{model_name}_similarity_plot.png"
    )
    fig.show()
    fig.waitforbuttonpress()
