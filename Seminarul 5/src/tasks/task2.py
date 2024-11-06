from numpy import ndarray
from sklearn.decomposition import TruncatedSVD


def task2(bow_matrix: ndarray, terms: ndarray):
    model, matrix = run(bow_matrix, 5)
    show(model, matrix, terms)


def run(bow_matrix: ndarray, topics: int):
    model = TruncatedSVD(n_components=topics, random_state=42)
    matrix = model.fit_transform(bow_matrix)

    return model, matrix


def show(model: TruncatedSVD, matrix: ndarray, terms: ndarray):
    print("SVD Topics")
    for i, topic in enumerate(model.components_):
        top_words = [terms[index] for index in topic.argsort()[-10:]]
        print(f"SVD Topic {i+1}: {', '.join(top_words)}")

    print("SVD Matrix")
    print(matrix)
