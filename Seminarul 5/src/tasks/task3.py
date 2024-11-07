from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.decomposition import NMF


def task3(bow_matrix: ndarray, terms: ndarray):
    model, matrix = run(bow_matrix, 5)
    show(model, matrix, terms)


def run(bow_matrix: ndarray, topics: int):
    model = NMF(n_components=topics, random_state=42)
    matrix = model.fit_transform(bow_matrix)

    return model, matrix


def show(model: NMF, matrix: ndarray, terms: ndarray):
    print("NMF Topics")
    for i, topic in enumerate(model.components_):
        top_words = [terms[index] for index in topic.argsort()[-10:]]
        print(f"NMF Topic {i+1}: {', '.join(top_words)}")

    print("NMF Matrix")
    print(matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        matrix[:, 0],
        matrix[:, 1],
        matrix[:, 2],
    )

    for i in range(matrix.shape[0]):
        ax.text(
            matrix[i, 0],
            matrix[i, 1],
            matrix[i, 2],
            f"NMF Topic {i + 1}",
            size=12,
            zorder=1,
        )

    plt.show()
