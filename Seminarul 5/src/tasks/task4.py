from numpy import ndarray
from sklearn.decomposition import LatentDirichletAllocation


def task4(bow_matrix: ndarray, terms: ndarray):
    model, matrix = run(bow_matrix, 5)
    show(model, matrix, terms)


def run(bow_matrix: ndarray, topics: int):
    model = LatentDirichletAllocation(n_components=topics, random_state=42)
    matrix = model.fit_transform(bow_matrix)

    return model, matrix


def show(model: LatentDirichletAllocation, matrix: ndarray, terms: ndarray):
    print("LDA Topics")
    for i, topic in enumerate(model.components_):
        top_words = [terms[index] for index in topic.argsort()[-10:]]
        print(f"LDA Topic {i+1}: {', '.join(top_words)}")

    print("LDA Matrix")
    print(matrix)
