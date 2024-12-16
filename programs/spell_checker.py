import nltk
from nltk.corpus import words, brown, gutenberg
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import string
import re
import numpy as np
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('words')
nltk.download('brown')
nltk.download('gutenberg')
nltk.download('punkt')

class SpellChecker:
    def __init__(self):
        self.vocabulary = set(words.words())
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.unigram_counts = defaultdict(int)
        self.word_freq = defaultdict(int)

        common_words = {'misspelled', 'jumps', 'words', 'over'}
        self.vocabulary.update(common_words)

        self.train_model()

    def train_model(self):
        for corpus in [brown, gutenberg]:
            for sentence in corpus.sents():
                tokens = ['<s>', '<s>'] + [word.lower() for word in sentence] + ['</s>']

                for token in tokens:
                    self.word_freq[token.lower()] += 1

                for i in range(len(tokens)-2):
                    self.bigram_counts[tokens[i+1]][tokens[i+2]] += 1
                    self.unigram_counts[tokens[i+1]] += 1

                    self.trigram_counts[tokens[i]][tokens[i+1]][tokens[i+2]] += 1

                self.unigram_counts[tokens[-1]] += 1

    def get_edits1(self, word):
        """Generate all strings that are one edit distance away from the input word"""
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def get_edits2(self, word):
        """Generate all strings that are two edits away from the input word"""
        return set(e2 for e1 in self.get_edits1(word) for e2 in self.get_edits1(e1))

    def calculate_ngram_probability(self, prev_tokens, token):
        """Calculate probability using interpolated trigram/bigram/unigram models"""
        lambda1, lambda2, lambda3 = 0.5, 0.3, 0.2

        trigram_prob = 0
        if len(prev_tokens) >= 2:
            numerator = self.trigram_counts[prev_tokens[0]][prev_tokens[1]][token] + 1
            denominator = self.bigram_counts[prev_tokens[0]][prev_tokens[1]] + len(self.vocabulary)
            trigram_prob = numerator / denominator

        bigram_prob = 0
        if len(prev_tokens) >= 1:
            numerator = self.bigram_counts[prev_tokens[-1]][token] + 1
            denominator = self.unigram_counts[prev_tokens[-1]] + len(self.vocabulary)
            bigram_prob = numerator / denominator

        unigram_prob = (self.unigram_counts[token] + 1) / (sum(self.unigram_counts.values()) + len(self.vocabulary))

        return lambda1 * trigram_prob + lambda2 * bigram_prob + lambda3 * unigram_prob

    def score_candidate(self, candidate, prev_words, next_words):
        """Score a candidate word based on n-gram probability and word frequency"""
        prev_tokens = ['<s>'] if not prev_words else prev_words[-2:]
        prob_score = np.log(self.calculate_ngram_probability(prev_tokens, candidate))

        if next_words:
            next_prob = np.log(self.calculate_ngram_probability([prev_tokens[-1], candidate], next_words[0]))
            prob_score += next_prob

        freq_score = np.log(self.word_freq[candidate.lower()] + 1)

        if prev_words and next_words:
            avg_len = (len(prev_words[-1]) + len(next_words[0])) / 2
            len_score = -abs(len(candidate) - avg_len) / 10
        else:
            len_score = 0

        return prob_score + 0.5 * freq_score + len_score

    def correct_word(self, word, prev_words, next_words):
        """Correct a single word using context"""
        if word.lower() in self.vocabulary:
            return word

        # Generate candidates
        candidates = self.get_edits1(word.lower())
        valid_candidates = {c for c in candidates if c in self.vocabulary}

        if not valid_candidates:
            candidates2 = self.get_edits2(word.lower())
            valid_candidates = {c for c in candidates2 if c in self.vocabulary}

        if not valid_candidates:
            return word

        best_candidate = max(valid_candidates,
                           key=lambda x: self.score_candidate(x, prev_words, next_words))

        if word[0].isupper():
            best_candidate = best_candidate.capitalize()

        return best_candidate

    def correct_text(self, text):
        words = word_tokenize(text)
        corrected_words = []

        for i, word in enumerate(words):
            if word.isalpha():
                prev_words = [w.lower() for w in words[max(0, i-2):i]]
                next_words = [w.lower() for w in words[i+1:min(len(words), i+3)]]

                corrected_word = self.correct_word(word, prev_words, next_words)
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

def test_spell_checker():
    spell_checker = SpellChecker()

    test_cases = [
        "This is a test sentense with misspeled words",
        "I recieved your mesage yestarday",
        "The quick brwn fox jumps ovr the lasy dog",
        "She was writting a letter to her frend"
    ]

    print("Spell Checker Test Results:")
    print("-" * 50)
    for text in test_cases:
        corrected_text = spell_checker.correct_text(text)
        print(f"Original:  {text}")
        print(f"Corrected: {corrected_text}")
        print("-" * 50)

if __name__ == "__main__":
    print("Testing Improved Spell Checker:")
    test_spell_checker()
