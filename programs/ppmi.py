import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import itertools

# Function to preprocess text (simple tokenization)
def preprocess(text):
    return text.lower().split()

# Function to build co-occurrence matrix
def build_cooccurrence_matrix(corpus, window_size=2):
    vocabulary = set()
    word_counts = Counter()
    cooccurrence_counts = defaultdict(lambda: defaultdict(int))

    # Tokenize sentences and build counts
    for sentence in corpus:
        tokens = preprocess(sentence)
        vocabulary.update(tokens)
        word_counts.update(tokens)

        for i, word in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    cooccurrence_counts[word][tokens[j]] += 1

    vocabulary = sorted(vocabulary)
    word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
    cooccurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))

    for word, neighbors in cooccurrence_counts.items():
        for neighbor, count in neighbors.items():
            cooccurrence_matrix[word_to_index[word]][word_to_index[neighbor]] = count

    return cooccurrence_matrix, word_to_index, vocabulary

# Function to compute PPMI matrix
def compute_ppmi_matrix(cooccurrence_matrix):
    total_sum = np.sum(cooccurrence_matrix)
    word_sum = np.sum(cooccurrence_matrix, axis=1)
    context_sum = np.sum(cooccurrence_matrix, axis=0)

    ppmi_matrix = np.zeros_like(cooccurrence_matrix)
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            p_wc = cooccurrence_matrix[i][j] / total_sum
            p_w = word_sum[i] / total_sum
            p_c = context_sum[j] / total_sum

            if p_wc > 0:
                ppmi = max(0, np.log2(p_wc / (p_w * p_c)))
                ppmi_matrix[i][j] = ppmi

    return ppmi_matrix

# Function to compute cosine similarity between two vectors
def cosine_similarity_vectors(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

# Function to compute cosine similarity between two words
def cosine_similarity_words(word1, word2, ppmi_matrix, word_to_index):
    if word1 not in word_to_index or word2 not in word_to_index:
        return 0.0
    index1 = word_to_index[word1]
    index2 = word_to_index[word2]
    return cosine_similarity_vectors(ppmi_matrix[index1], ppmi_matrix[index2])

# Example usage
if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat",
        "the dog barked at the cat",
        "the mouse ran across the room",
        "the cat chased the mouse"
    ]

    # Build co-occurrence and PPMI matrices
    cooccurrence_matrix, word_to_index, vocabulary = build_cooccurrence_matrix(corpus, window_size=2)
    ppmi_matrix = compute_ppmi_matrix(cooccurrence_matrix)

    print("Vocabulary:")
    print(vocabulary)
    # print("\nPPMI Matrix:")
    # print(ppmi_matrix)

    # Cosine similarity between words
    word1 = "cat"
    word2 = "dog"
    similarity = cosine_similarity_words(word1, word2, ppmi_matrix, word_to_index)
    print(f"\nCosine Similarity between '{word1}' and '{word2}': {similarity:.4f}")

    # Cosine similarity between two documents
    doc1_vector = np.sum(ppmi_matrix, axis=0)  # Sum word vectors for a document
    doc2_vector = np.sum(ppmi_matrix, axis=1)  # Sum word vectors for a second document
    doc_similarity = cosine_similarity_vectors(doc1_vector, doc2_vector)
    print(f"\nCosine Similarity between Document 1 and Document 2: {doc_similarity:.4f}")
