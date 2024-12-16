from collections import defaultdict
import re

# Function to preprocess text (simple tokenization)
def preprocess(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    return text.split()

# Function to calculate bigram probabilities
def calculate_bigrams(corpus):
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(lambda: defaultdict(int))

    # Tokenize corpus and calculate counts
    for sentence in corpus:
        tokens = ['<s>'] + preprocess(sentence) + ['</s>']
        for i in range(len(tokens) - 1):
            unigram_counts[tokens[i]] += 1
            bigram_counts[tokens[i]][tokens[i + 1]] += 1
        unigram_counts[tokens[-1]] += 1  # Count </s> as a unigram

    # Calculate bigram probabilities
    bigram_probs = defaultdict(dict)
    for w1 in bigram_counts:
        for w2 in bigram_counts[w1]:
            bigram_probs[w1][w2] = bigram_counts[w1][w2] / unigram_counts[w1]

    return bigram_probs

# Function to calculate sentence probability using bigram model
def sentence_probability(sentence, bigram_probs):
    tokens = ['<s>'] + preprocess(sentence) + ['</s>']
    prob = 1.0

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w2 in bigram_probs.get(w1, {}):
            prob *= bigram_probs[w1][w2]
        else:
            prob *= 0  # If bigram doesn't exist, probability is 0

    return prob

# Example usage
if __name__ == "__main__":
    corpus = [
        "The cat sat on the mat",
        "The cat ate the mouse",
        "The dog barked loudly"
    ]

    # Calculate bigram probabilities
    bigram_probs = calculate_bigrams(corpus)

    # Print bigram probabilities
    print("Bigram Probabilities:")
    for w1 in bigram_probs:
        for w2 in bigram_probs[w1]:
            print(f"P({w2} | {w1}) = {bigram_probs[w1][w2]:.4f}")

    # Test sentence probability
    test_sentence = "The cat sat"
    prob = sentence_probability(test_sentence, bigram_probs)
    print(f"\nProbability of sentence '{test_sentence}': {prob:.8f}")
