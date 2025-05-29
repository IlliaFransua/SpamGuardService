import os
import joblib
import numpy as np
from typing import List
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


class SpamDetectionModel:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

    def __init__(self, threshold: float = 0.69, n_features: int = 2**20):
        """
        Initialize the spam detection model for online learning.
        Uses HashingVectorizer and SGDClassifier for infinite incremental training.
        """
        self.threshold = threshold
        self.classes = np.array([0, 1])

        os.makedirs(self.MODEL_DIR, exist_ok=True)

        # Initialize or load classifier
        if os.path.exists(self.MODEL_FILE):
            self.classifier = joblib.load(self.MODEL_FILE)
        else:
            self.classifier = SGDClassifier(loss='log_loss', max_iter=1, tol=None)

        # HashingVectorizer is stateless, no need to load/save
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            ngram_range=(1, 2)
        )

    def train(self, spam_texts: List[str], non_spam_texts: List[str]) -> None:
        """
        Incrementally train on new batch of data.
        This can be called indefinitely; the model will continuously learn.
        """
        new_texts = spam_texts + non_spam_texts
        new_labels = [1] * len(spam_texts) + [0] * len(non_spam_texts)

        if not new_texts:
            raise ValueError("Training data cannot be empty")

        X = self.vectorizer.transform(new_texts)
        y = np.array(new_labels)

        # Online update
        self.classifier.partial_fit(X, y, classes=self.classes)
        joblib.dump(self.classifier, self.MODEL_FILE)

    def predict_spam_chance(self, text: str) -> float:
        """
        Return probability of message being spam (0.0 to 1.0).
        """
        try:
            X = self.vectorizer.transform([text])
            return self.classifier.predict_proba(X)[0, 1]
        except NotFittedError:
            raise RuntimeError("Model is not trained yet. Call `train()` first.")

    def predict(self, text: str) -> bool:
        """
        Predict whether a message is spam based on threshold.
        """
        spam_prob = self.predict_spam_chance(text)
        print(f"Spam probability: {spam_prob:.4f}: {spam_prob >= self.threshold}: {text}")
        return spam_prob >= self.threshold

    def test(self, spam_texts: List[str], non_spam_texts: List[str]) -> float:
        """
        Compute accuracy on provided test data.
        """
        texts = spam_texts + non_spam_texts
        labels = [1] * len(spam_texts) + [0] * len(non_spam_texts)

        if not texts:
            raise ValueError("Test data cannot be empty")

        X = self.vectorizer.transform(texts)
        return self.classifier.score(X, labels)

    def save_all(self) -> None:
        """
        Save the classifier to disk.
        """
        joblib.dump(self.classifier, self.MODEL_FILE)
