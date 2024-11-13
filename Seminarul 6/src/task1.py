from numpy import array, ndarray, stack
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from util import (
    make_dirs,
    pick_dimensions,
    pick_words,
    read,
    scale_between_0_and_1,
    vectorize,
)


def task1():
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


def plot_random_dimensions(
    model_name: str,
    words: list[str],
    vectorized_words: list[ndarray],
    dimensions: list[int],
):
    plot_common(model_name, "Random_dimensions", words, vectorized_words, dimensions)


def plot_pca(model_name: str, words: list[str], vectorized_words: list[ndarray]):
    pca = PCA(n_components=3)
    dimensions = range(0, 3)
    stacked_vectorized_words = stack(vectorized_words)

    stacked_decomposed_vectorized_words = pca.fit_transform(stacked_vectorized_words)
    decomposed_vectorized_words = [row for row in stacked_decomposed_vectorized_words]

    plot_common(model_name, "PCA", words, decomposed_vectorized_words, dimensions)


def plot_tsne(model_name: str, words: list[str], vectorized_words: list[ndarray]):
    tsne = TSNE(n_components=3, perplexity=6, random_state=0)
    dimensions = range(0, 3)
    stacked_vectorized_words = stack(vectorized_words)

    stacked_decomposed_vectorized_words = tsne.fit_transform(stacked_vectorized_words)
    decomposed_vectorized_words = [row for row in stacked_decomposed_vectorized_words]

    plot_common(model_name, "t-SNE", words, decomposed_vectorized_words, dimensions)


def plot_common(
    model_name: str,
    method_name: str,
    words: list[str],
    vectorized_words: list[ndarray],
    dimensions: list[int],
):
    xs, ys, zs = tuple(
        array([vector[dimension] for vector in vectorized_words])
        for dimension in dimensions
    )
    reds = scale_between_0_and_1(xs)
    blues = scale_between_0_and_1(ys)
    greens = scale_between_0_and_1(zs)
    colors = stack((reds, blues, greens), axis=-1)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xs, ys, zs, marker="o", c=colors)

    for i, word in enumerate(words):
        if i >= len(vectorized_words):
            break

        ax.text(xs[i], ys[i], zs[i], word, size=12, zorder=1, color="k")

    ax.set_xlabel(f"Dimension {dimensions[0]}")
    ax.set_ylabel(f"Dimension {dimensions[1]}")
    ax.set_zlabel(f"Dimension {dimensions[2]}")
    ax.set_title(f"{method_name} plot")

    fig.savefig(
        f"./outputs/{method_name}_plot(model: {model_name}, dimensions: {", ".join([str(dimension) for dimension in dimensions])}).png"
    )
    fig.show()
    fig.waitforbuttonpress()
