from spam_detection_model import SpamDetectionModel
from archive_spam_detection_model import ArchiveSpamDetectionModel
import pandas as pd
from typing import List, Tuple


def prepare_dataset() -> Tuple[List[str], List[int]]:
    test_df = pd.read_csv('dataset/spam_text_test_dataset.csv')
    test_df = test_df.dropna(subset=["text"])

    spam_test_texts = test_df[test_df["label"] == 1]["text"].tolist()
    non_spam_test_texts = test_df[test_df["label"] == 0]["text"].tolist()

    test_texts = spam_test_texts + non_spam_test_texts
    test_labels = [1] * len(spam_test_texts) + [0] * len(non_spam_test_texts)

    return test_texts, test_labels


def test_archive_model() -> None:
    spam_detector = ArchiveSpamDetectionModel()

    test_texts, test_labels = prepare_dataset()

    correct_predictions = 0

    print("Testing Archive Spam Detection Model...")

    for text, true_label in zip(test_texts, test_labels):
        result = spam_detector.predict(text)
        predicted_label = 1 if result else 0

        if predicted_label == true_label:
            correct_predictions += 1

        # print(f"Text: {text}\nPredicted as: {'Spam' if predicted_label else 'Not Spam'} | True label: {'Spam' if true_label == 1 else 'Not Spam'}\n")

    total_tests = len(test_texts)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")

def test_current_model() -> None:
    spam_detector = SpamDetectionModel()

    test_texts, test_labels = prepare_dataset()

    correct_predictions = 0

    print("Testing Current Spam Detection Model...")

    for text, true_label in zip(test_texts, test_labels):
        result = spam_detector.predict(text)
        predicted_label = 1 if result else 0

        if predicted_label == true_label:
            correct_predictions += 1

        # print(f"Text: {text}\nPredicted as: {'Spam' if predicted_label else 'Not Spam'} | True label: {'Spam' if true_label == 1 else 'Not Spam'}\n")

    total_tests = len(test_texts)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_archive_model()
    print("===========")
    test_current_model()
