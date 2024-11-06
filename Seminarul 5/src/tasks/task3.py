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
