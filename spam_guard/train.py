from typing import Iterator, List

from model import SpamDetectionModel

def read_in_batches(file_path: str, batch_size: int) -> Iterator[List[str]]:
    """
    Generator that yields lines from a text file in batches.
    """
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

good_file = '/Users/illiafransua/Documents/Projects/SpamGuardService/spam_guard/profanity_list/false_list.txt'
bad_file = '/Users/illiafransua/Documents/Projects/SpamGuardService/spam_guard/profanity_list/true_list.txt'
batch_size = 500

model = SpamDetectionModel(threshold=0.6)

for good_batch, bad_batch in zip(read_in_batches(good_file, batch_size),
                                 read_in_batches(bad_file, batch_size)):
    model.train(bad_batch, good_batch)
    print(f"Trained on batch of {len(bad_batch)} spam + {len(good_batch)} non-spam")

model.save_all()
print("Training complete.")
