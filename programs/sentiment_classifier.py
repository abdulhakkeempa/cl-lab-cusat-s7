from nltk.corpus import movie_reviews

import numpy as np

import re
import string
import numpy as np
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.model_selection import train_test_split

class NaiveBayesSentimentClassifier:
    def __init__(self, k=1.0):
        self.k = k  # Smoothing parameter
        self.word_counts = {'pos': defaultdict(int), 'neg': defaultdict(int)}
        self.class_counts = {'pos': 0, 'neg': 0}
        self.vocabulary = set()

    def preprocess(self, text):
        """Preprocess the text by converting to lowercase and removing punctuation"""
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', '', text)
        return word_tokenize(text)

    def train(self, X_train, y_train):
        """Train the classifier on the given data"""
        for text, label in zip(X_train, y_train):
            words = self.preprocess(text)
            self.class_counts[label] += 1

            for word in words:
                self.word_counts[label][word] += 1
                self.vocabulary.add(word)

    def calculate_probability(self, text, label):
        """Calculate P(text|label) using the Naive Bayes assumption"""
        words = self.preprocess(text)
        log_prob = np.log(self.class_counts[label] / sum(self.class_counts.values()))

        vocab_size = len(self.vocabulary)
        total_words = sum(self.word_counts[label].values())

        for word in words:
            count = self.word_counts[label].get(word, 0)
            prob = (count + self.k) / (total_words + self.k * vocab_size)
            log_prob += np.log(prob)

        return log_prob

    def predict(self, text):
        """Predict the sentiment of the given text"""
        pos_prob = self.calculate_probability(text, 'pos')
        neg_prob = self.calculate_probability(text, 'neg')

        return 'pos' if pos_prob > neg_prob else 'neg'

    def evaluate(self, X_test, y_test):
        """Evaluate the classifier on test data"""
        correct = 0
        total = len(X_test)

        for text, true_label in zip(X_test, y_test):
            pred_label = self.predict(text)
            if pred_label == true_label:
                correct += 1

        return correct / total

def test_sentiment_classifier():
    documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]

    np.random.shuffle(documents)

    texts = [' '.join(doc) for doc, category in documents]
    labels = [category for doc, category in documents]

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    k_values = [0.25, 0.75, 1.0]

    for k in k_values:
        classifier = NaiveBayesSentimentClassifier(k=k)
        classifier.train(X_train, y_train)
        accuracy = classifier.evaluate(X_test, y_test)
        print(f"Accuracy with k={k}: {accuracy:.4f}")

test_sentiment_classifier()
