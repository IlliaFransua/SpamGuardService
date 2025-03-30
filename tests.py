from spam_detection_model import SpamDetectionModel

if __name__ == "__main__":
    spam_detector = SpamDetectionModel()

    test_spam = [
        "Congratulations! You've won a lottery. Claim your prize now!",
        "Limited time offer: Buy now and save big!"
    ]
    test_non_spam = [
        "Hi, how are you doing today?",
        "Let's meet for lunch tomorrow."
    ]

    print("Testing on spam lines:")
    for text in test_spam:
        result = spam_detector.predict(text)
        print(f"Text: {text}\nPredicted as: {'Spam' if result == 1 else 'Not Spam'}\n")

    print("Testing on non-spam strings:")
    for text in test_non_spam:
        result = spam_detector.predict(text)
        print(f"Text: {text}\nPredicted as: {'Spam' if result == 1 else 'Not Spam'}\n")
