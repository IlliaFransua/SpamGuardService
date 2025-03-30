import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from typing import List
import pandas as pd


class SpamDetectionModel:
    MODEL_FILE = "model_training/spam_model.pkl"
    DATA_FILE = "model_training/spam_data.npz"
    VECTOR_FILE = "model_training/vectorizer.pkl"

    def __init__(self):
        self.texts = []
        self.labels = []

        self.check_and_create_files()

        self.vectorizer = joblib.load(self.VECTOR_FILE)
        self.classifier = joblib.load(self.MODEL_FILE)
        self.load_training_data()

    def check_and_create_files(self):
        if not os.path.exists(self.VECTOR_FILE):
            # print(f"File {self.VECTOR_FILE} not found. Creating new")
            joblib.dump(CountVectorizer(), self.VECTOR_FILE)

        if not os.path.exists(self.MODEL_FILE):
            # print(f"File {self.MODEL_FILE} not found. Creating new")
            joblib.dump(SGDClassifier(loss='log_loss'), self.MODEL_FILE)

        if not os.path.exists(self.DATA_FILE):
            # print(f"File {self.DATA_FILE} not found. Creating new")
            self.save_training_data()

    def load_training_data(self):
        if os.path.exists(self.DATA_FILE) and os.path.getsize(self.DATA_FILE) > 0:
            data = np.load(self.DATA_FILE, allow_pickle=True)
            self.texts = data['texts'].tolist()
            self.labels = data['labels'].tolist()
        else:
            self.texts = []
            self.labels = []

    def save_training_data(self):
        np.savez(self.DATA_FILE, texts=self.texts, labels=self.labels)

    def train(self, spam_texts: List[str], non_spam_texts: List[str]):
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

    def predict(self, text: str) -> bool:
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]

    def test(self, spam_texts: List[str], non_spam_texts: List[str]) -> float:
        texts = spam_texts + non_spam_texts
        labels = [1] * len(spam_texts) + [0] * len(non_spam_texts)

        if not texts:
            raise ValueError("Test data cannot be empty")

        X = self.vectorizer.transform(texts)
        return self.classifier.score(X, labels)

    def save_all(self):
        joblib.dump(self.vectorizer, self.VECTOR_FILE)
        joblib.dump(self.classifier, self.MODEL_FILE)
        self.save_training_data()
