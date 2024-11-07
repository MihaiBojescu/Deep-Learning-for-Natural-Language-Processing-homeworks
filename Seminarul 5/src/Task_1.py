# Import these modules
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import os
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")


# The following function is used to remove zero variance columns (terms that don't appear in any document)
def removeZeroVarianceColumns(tfidf_matrix):
    # Check for columns with zero variance (e.g. terms that appear in no document)
    nonzero_variance_columns = np.array(tfidf_matrix.sum(axis=0)).flatten() > 0

    # Filter the matrix to retain only the columns with non-zero variance
    return tfidf_matrix[:, nonzero_variance_columns]


# The following function is used to perform TF-IDF vectorization
def tfIdfVectorization(document):
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')  # Removing common stop words
    tfidf_matrix = vectorizer.fit_transform(document)

    # Get the list of terms from the vectorizer
    terms = vectorizer.get_feature_names_out()

    # Ensure no empty/zero variance columns exist (e.g. terms with no occurrence in the corpus)
    tfidf_matrix = removeZeroVarianceColumns(tfidf_matrix)

    return tfidf_matrix, terms


# The following function is used to perform Latent Semantic Analysis using SVD
def performLSA(tfidf_matrix, num_topics=5):
    # Apply TruncatedSVD for LSA with the specified number of topics (components)
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)

    # Fit the LSA model on the TF-IDF matrix
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)

    # Handle NaN or infinite values in the SVD result
    if np.any(np.isnan(lsa_matrix)) or np.any(np.isinf(lsa_matrix)):
        print("LSA matrix contains NaN or infinite values!")

    return lsa_model, lsa_matrix


# The following function is used to perform Latent Semantic Analysis using NMF (Non-negative Matrix Factorization)
def performNMF(tfidf_matrix, num_topics=5):
    # Make sure that n_components is no greater than min(n_samples, n_features)
    num_topics = min(num_topics, min(tfidf_matrix.shape))  # Limit to min(n_samples, n_features)

    # Apply NMF for topic modeling with the specified number of topics (components)
    nmf_model = NMF(n_components=num_topics, random_state=42)  # Use n_components or n_topics based on version

    # Fit the NMF model on the TF-IDF matrix
    nmf_matrix = nmf_model.fit_transform(tfidf_matrix)

    return nmf_model, nmf_matrix


# The following function is used to perform Latent Dirichlet Allocation (LDA)
def perform_lda(tfidf_matrix, num_topics=5):
    # Apply LDA for topic modeling with the specified number of topics
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

    # Fit the LDA model on the TF-IDF matrix
    lda_matrix = lda_model.fit_transform(tfidf_matrix)

    return lda_model, lda_matrix


# The following function is used to display topics and the associated terms
def displayTopics(nmf_model, terms, top_n=5):
    print("\nTopics (components):")
    for idx, topic in enumerate(nmf_model.components_):
        print(f"Topic {idx + 1}:")

        # Sort the terms by their importance (highest weight first) in the topic
        sorted_terms = [terms[i] for i in topic.argsort()[-top_n:]]  # Show top 'top_n' terms for each topic
        print(" ".join(sorted_terms))
        print()


# The following function is used to display the Document-Topic relationship matrix
def displayDocumentTopicMatrix(nmf_matrix):
    print("\nDocument-Topic matrix:")
    print(nmf_matrix)


def plot3D(lsa_matrix, num_topics=5):
    # Assuming lsa_matrix is the result of the LSA transformation (after dimensionality reduction)
    # We will visualize the first three components (topics) in a 3D space.

    # Only take the first 3 components for the plot
    X = lsa_matrix[:, :3]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='o')

    ax.set_xlabel('Topic 1')
    ax.set_ylabel('Topic 2')
    ax.set_zlabel('Topic 3')

    ax.set_title('3D Visualization of Document-Topic Relationships')

    plt.show()


def Task_1():
    lemmatizer = WordNetLemmatizer()

    # Get the list of all files and directories
    path = "../data/raw"
    dir_list = os.listdir(path)

    directory_path = "../data/lemmatized/"

    # Create the directory if it does not exist
    os.makedirs(directory_path, exist_ok=True)

    for file in dir_list:
        with open("../data/raw/" + file, "r", encoding="utf-8") as f:
            with open("../data/lemmatized/" + file, "w", encoding="utf-8") as f_out:
                for line in f:
                    lemmatized_words: dict[str, int] = {}
                    for word in line.split(" "):

                        match = re.match(r"(\w+)", word)

                        if match is not None and len(match.groups()) == 1:
                            new_word = lemmatizer.lemmatize(match[0])
                            # print(word, new_word)

                            lemmatized_words[new_word] = (
                                lemmatized_words[new_word] + 1
                                if new_word in lemmatized_words
                                else 1
                            )

                            f_out.write("(" + str(match[0]) + ", " + str(new_word) + ")\n")

    '''
    print("rocks :", lemmatizer.lemmatize("rocks"))
    print("corpora :", lemmatizer.lemmatize("corpora"))
    
    # a denotes adjective in "pos"
    print("better :", lemmatizer.lemmatize("better", pos="a"))
    '''


# Latent Semantic Analysis (LSA) is a technique used to reduce the dimensionality of the TF-IDF matrix and uncover the
# latent semantic structure of the text data. One of the most common ways to perform LSA is by using Singular Value
# Decomposition (SVD), which decomposes the TF-IDF matrix into a set of matrices that capture the most significant
# patterns in the data.

def Task_2():
    # Get the list of all files and directories
    path = "../data/raw"
    dir_list = os.listdir(path)

    for file in dir_list:
        document = []

        with open("../data/raw/" + file, "r", encoding="utf-8") as f:
            content = f.read().split("\n")

            # Step 1: Define the text document
            for element in content:
                document.append(element)

            # Step 2: Perform TF-IDF Vectorization
            tfidf_matrix, terms = tfIdfVectorization(document)

            # Step 3: Perform Latent Semantic Analysis (LSA) with 5 topics
            num_topics = 5
            lsa_model, lsa_matrix = performLSA(tfidf_matrix, num_topics=num_topics)

            # Step 4: Display topics (terms associated with each topic)
            displayTopics(lsa_model, terms, top_n=5)

            # Step 5: Display the Document-Topic matrix
            displayDocumentTopicMatrix(lsa_matrix)

            # Step 6: Plot the 3D representation of the Document-Topic matrix
            plot3D(lsa_matrix)

        print("--------------------------------------------------\n\n")


# The negative values in the document-topic matrix can occur due to the nature of the Singular Value Decomposition
# (SVD) algorithm. SVD works by approximating the original matrix with lower-rank matrices, and sometimes the matrix
# components (singular values, topic loadings, etc.) can have negative values. This is not unusual, especially for
# sparse matrices and the way topics are represented.
#
# However, typically, document-topic matrices should be non-negative because they represent the association between
# documents and topics, which should logically be positive (indicating the strength of association). Negative values
# can appear due to how the algorithm factors the data, and they usually occur when the factorization process includes
# subtraction of components in the SVD decomposition.
#
# To fix this, we can apply non-negative matrix factorization (NMF) instead of SVD. NMF ensures that the resulting
# matrix contains only non-negative values. In the context of Latent Semantic Analysis (LSA), NMF is a suitable
# alternative for topic modeling that guarantees non-negative results.
def Task_3():
    # Get the list of all files and directories
    path = "../data/raw"
    dir_list = os.listdir(path)

    for file in dir_list:
        document = []

        with open("../data/raw/" + file, "r", encoding="utf-8") as f:
            content = f.read().split("\n")

            # Step 1: Define the text document
            for element in content:
                document.append(element)

            # Step 2: Perform TF-IDF Vectorization
            tfidf_matrix, terms = tfIdfVectorization(document)

            # Step 3: Perform Latent Semantic Analysis (LSA) with 5 topics
            num_topics = 5
            nmf_model, nmf_matrix = performNMF(tfidf_matrix, num_topics=num_topics)

            # Step 4: Display topics (terms associated with each topic)
            displayTopics(nmf_model, terms, top_n=5)

            # Step 5: Display the Document-Topic matrix
            displayDocumentTopicMatrix(nmf_matrix)

        print("--------------------------------------------------\n\n")


# Latent Dirichlet Allocation (LDA) is a popular probabilistic model used for topic modeling. LDA works differently
# than NMF or SVD, as it is based on a generative process where each document is considered as a mixture of topics,
# and each topic is characterized by a distribution over words.
def Task_4():
    # Get the list of all files and directories
    path = "../data/raw"
    dir_list = os.listdir(path)

    for file in dir_list:
        document = []

        with open("../data/raw/" + file, "r", encoding="utf-8") as f:
            content = f.read().split("\n")

            # Step 1: Define the text document
            for element in content:
                document.append(element)

            # Step 2: Perform TF-IDF Vectorization
            tfidf_matrix, terms = tfIdfVectorization(document)

            # Step 3: Perform Latent Dirichlet Allocation (LDA) with 5 topics
            num_topics = 5
            lda_model, lda_matrix = perform_lda(tfidf_matrix, num_topics=num_topics)

            # Step 4: Display topics (terms associated with each topic)
            displayTopics(lda_model, terms, top_n=5)

            # Step 5: Display the Document-Topic matrix
            displayDocumentTopicMatrix(lda_matrix)

        print("--------------------------------------------------\n\n")


if __name__ == "__main__":
    # Task_1()
    Task_2()
    # Task_3()
    # Task_4()
