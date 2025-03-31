import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from typing import List
import pandas as pd


class SpamDetectionModel:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model_training")

    MODEL_FILE = os.path.join(MODEL_DIR, "spam_model.pkl")
    DATA_FILE = os.path.join(MODEL_DIR, "spam_data.npz")
    VECTOR_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")

    def __init__(self):
        """
        Initialize the spam detection model.
        Loads the model, vectorizer, and training data.
        """
        self.texts = []
        self.labels = []
        self.threshold = 0.6

        self.check_and_create_files()

        self.vectorizer = joblib.load(self.VECTOR_FILE)
        self.classifier = joblib.load(self.MODEL_FILE)
        self.load_training_data()

    def check_and_create_files(self):
        """
        Ensure all necessary files exist.
        If files do not exist, create and save default ones.
        """
        if not os.path.exists(self.VECTOR_FILE):
            joblib.dump(CountVectorizer(), self.VECTOR_FILE)

        if not os.path.exists(self.MODEL_FILE):
            joblib.dump(SGDClassifier(loss='log_loss'), self.MODEL_FILE)

        if not os.path.exists(self.DATA_FILE):
            self.save_training_data()

    def load_training_data(self):
        """
        Load training data from file if it exists.
        Otherwise, initialize empty lists for texts and labels.
        """
        if os.path.exists(self.DATA_FILE) and os.path.getsize(self.DATA_FILE) > 0:
            data = np.load(self.DATA_FILE, allow_pickle=True)
            self.texts = data['texts'].tolist()
            self.labels = data['labels'].tolist()
        else:
            self.texts = []
            self.labels = []

    def save_training_data(self):
        """
        Save current training data (texts and labels) to file.
        """
        np.savez(self.DATA_FILE, texts=self.texts, labels=self.labels)

    def train(self, spam_texts: List[str], non_spam_texts: List[str]):
        """
        Train the model with new spam and non-spam messages.
        Updates the training dataset and retrains the classifier.

        :param spam_texts: List of spam messages.
        :param non_spam_texts: List of non-spam messages.
        """
        new_texts = spam_texts + non_spam_texts
        new_labels = [1] * len(spam_texts) + [0] * len(non_spam_texts)

        if not new_texts:
            raise ValueError("Training data cannot be empty")

        self.texts.extend(new_texts)
        self.labels.extend(new_labels)
        self.save_training_data()

        X_all = self.vectorizer.fit_transform(self.texts)
        y_all = np.array(self.labels)
        joblib.dump(self.vectorizer, self.VECTOR_FILE)

        self.classifier.partial_fit(X_all, y_all, classes=np.array([0, 1]))
        joblib.dump(self.classifier, self.MODEL_FILE)

    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        :param x: Input value.
        :return: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def predict(self, text: str) -> bool:
        """
        Predict whether a message is spam.
        Uses the trained classifier to determine spam probability.

        :param text: Input message.
        :return: True if spam, False otherwise.
        """
        X = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(X)
        spam_probability = probabilities[0][1]
        spam_probability = self.sigmoid(spam_probability)
        print(spam_probability)
        return spam_probability >= self.threshold

    def test(self, spam_texts: List[str], non_spam_texts: List[str]) -> float:
        """
        Evaluate the model on test data.

        :param spam_texts: List of spam messages.
        :param non_spam_texts: List of non-spam messages.
        :return: Model accuracy on test data.
        """
        texts = spam_texts + non_spam_texts
        labels = [1] * len(spam_texts) + [0] * len(non_spam_texts)

        if not texts:
            raise ValueError("Test data cannot be empty")

        X = self.vectorizer.transform(texts)
        return self.classifier.score(X, labels)

    def save_all(self):
        """
        Save the model, vectorizer, and training data.
        """
        joblib.dump(self.vectorizer, self.VECTOR_FILE)
        joblib.dump(self.classifier, self.MODEL_FILE)
        self.save_training_data()
