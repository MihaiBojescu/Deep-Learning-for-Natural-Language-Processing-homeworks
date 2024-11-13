from numpy import array, ndarray, stack
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from util import scale_between_0_and_1


def plot_random_dimensions(
    model_name: str,
    words: list[str],
    vectorized_words: list[ndarray],
    dimensions: list[int],
):
    xs, ys, zs, reds, greens, blues = tuple(
        array([vector[dimension] for vector in vectorized_words])
        for dimension in dimensions
    )
    reds = scale_between_0_and_1(reds)
    blues = scale_between_0_and_1(blues)
    greens = scale_between_0_and_1(greens)
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
    ax.set_title("Random dimensions plot")

    fig.savefig(
        f"./outputs/random_dimensions_plot(model: {model_name}, dimensions: {", ".join([str(dimension) for dimension in dimensions])}).png"
    )
    fig.show()
    fig.waitforbuttonpress()


def plot_pca(model_name: str, words: list[str], vectorized_words: list[ndarray]):
    pca = PCA(n_components=6)
    dimensions = range(0, 6)
    stacked_vectorized_words = stack(vectorized_words)

    pca.fit(stacked_vectorized_words)

    stacked_decomposed_vectorized_words = pca.transform(stacked_vectorized_words)
    decomposed_vectorized_words = [row for row in stacked_decomposed_vectorized_words]

    xs, ys, zs, reds, greens, blues = tuple(
        array([vector[dimension] for vector in decomposed_vectorized_words])
        for dimension in dimensions
    )
    reds = scale_between_0_and_1(reds)
    blues = scale_between_0_and_1(blues)
    greens = scale_between_0_and_1(greens)
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
    ax.set_title("PCA plot")

    fig.savefig(
        f"./outputs/PCA_plot(model: {model_name}, dimensions: {", ".join([str(dimension) for dimension in dimensions])}).png"
    )
    fig.show()
    fig.waitforbuttonpress()
