import re
from collections import defaultdict, Counter
import math

# Training Data: Provided sentences with sense labels
train_data = [
    ("I love fish. The smoked bass fish was delicious.", "fish"),
    ("The bass fish swam along the line.", "fish"),
    ("He hauled in a big catch of smoked bass fish.", "fish"),
    ("The bass guitar player played a smooth jazz line.", "guitar"),
]

# Preprocess the sentences: tokenize and clean
def preprocess(sentence):
    return re.findall(r'\b\w+\b', sentence.lower())  # Lowercase and tokenize

# Build vocabulary and class-wise word counts
def train_naive_bayes(train_data):
    vocab = set()
    word_counts = defaultdict(Counter)  # Word counts per class
    class_counts = defaultdict(int)     # Count of sentences per class

    for sentence, label in train_data:
        tokens = preprocess(sentence)
        vocab.update(tokens)
        word_counts[label].update(tokens)
        class_counts[label] += 1

    return vocab, word_counts, class_counts

# Calculate log probabilities using Add-1 Smoothing
def calculate_log_probabilities(vocab, word_counts, class_counts):
    total_classes = sum(class_counts.values())
    log_probs = {}
    total_vocab_size = len(vocab)

    for label in class_counts:
        log_probs[label] = {
            'class_log_prob': math.log(class_counts[label] / total_classes),
            'word_log_probs': {}
        }
        total_words = sum(word_counts[label].values())

        for word in vocab:
            word_freq = word_counts[label][word] + 1  # Add-1 smoothing
            log_probs[label]['word_log_probs'][word] = math.log(word_freq / (total_words + total_vocab_size))

    return log_probs

# Predict the sense of the test word based on the test sentence
def predict(test_sentence, target_word, log_probs, vocab):
    tokens = preprocess(test_sentence)
    best_label = None
    best_log_prob = float('-inf')

    # Check probabilities for each class
    for label in log_probs:
        total_log_prob = log_probs[label]['class_log_prob']
        for word in tokens:
            if word in vocab:
                total_log_prob += log_probs[label]['word_log_probs'].get(word, 0)

        # Update best label if higher probability is found
        if total_log_prob > best_log_prob:
            best_log_prob = total_log_prob
            best_label = label

    return best_label

# Main function
def main():
    # Train the Naive Bayes model
    vocab, word_counts, class_counts = train_naive_bayes(train_data)
    log_probs = calculate_log_probabilities(vocab, word_counts, class_counts)

    # Test sentence
    test_sentence = "He loves jazz. The bass line provided the foundation for the guitar solo in the jazz piece"
    test_word = "bass"

    # Predict the sense of 'bass'
    predicted_sense = predict(test_sentence, test_word, log_probs, vocab)
    print(f"Test sentence: {test_sentence}")
    print(f"Test word: {test_word}")
    print(f"Output: {predicted_sense}")

if __name__ == "__main__":
    main()
