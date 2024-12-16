import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute TF-IDF matrix
def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer

# Function to calculate cosine similarity between two documents
def cosine_similarity_docs(tfidf_matrix, doc_index1, doc_index2):
    return cosine_similarity(tfidf_matrix[doc_index1], tfidf_matrix[doc_index2])[0][0]

# Function to calculate cosine similarity between two words
def cosine_similarity_words(word1, word2, vectorizer, feature_names, tfidf_matrix):
    if word1 not in feature_names or word2 not in feature_names:
        return 0.0

    word_index1 = np.where(feature_names == word1)[0][0]
    word_index2 = np.where(feature_names == word2)[0][0]

    word_vector1 = tfidf_matrix[:, word_index1].toarray().flatten()
    word_vector2 = tfidf_matrix[:, word_index2].toarray().flatten()

    return cosine_similarity(word_vector1.reshape(1, -1), word_vector2.reshape(1, -1))[0][0]

# Example usage
if __name__ == "__main__":
    documents = [
        "The cat sat on the mat",
        "The dog barked at the cat",
        "The mouse ran across the room",
        "The cat chased the mouse"
    ]

    # Compute TF-IDF matrix
    tfidf_matrix, feature_names, vectorizer = compute_tfidf(documents)
    print("TF-IDF Matrix:")
    print(tfidf_matrix.toarray())
    print("\nFeature Names:")
    print(feature_names)

    # Calculate cosine similarity between documents
    doc_index1 = 0  # First document
    doc_index2 = 1  # Second document
    similarity = cosine_similarity_docs(tfidf_matrix, doc_index1, doc_index2)
    print(f"\nCosine Similarity between Document {doc_index1} and Document {doc_index2}: {similarity:.4f}")

    # Calculate cosine similarity between words
    word1 = "cat"
    word2 = "dog"
    word_similarity = cosine_similarity_words(word1, word2, vectorizer, feature_names, tfidf_matrix)
    print(f"\nCosine Similarity between words '{word1}' and '{word2}': {word_similarity:.4f}")
