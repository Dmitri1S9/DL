"""
Скачивает LJSpeech с HuggingFace и делает train/test split.
LJSpeech: 13100 пар текст-аудио, один голос (женский), ~2.6GB.
"""

from datasets import load_dataset
from pathlib import Path

DATA_DIR = "data"

def download_ljspeech():
    print("Downloading LJSpeech from HuggingFace...")
    print("(~2.6GB, займёт несколько минут)\n")

    dataset = load_dataset("keithito/lj_speech", split="train", cache_dir=DATA_DIR)

    print(f"Total samples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample:\n  text: {dataset[0]['normalized_text']}")
    print(f"  audio: {dataset[0]['audio']['sampling_rate']} Hz\n")

    # Стандартный split: последние 500 = test
    test_size  = 500
    train_size = len(dataset) - test_size

    train_set = dataset.select(range(train_size))
    test_set  = dataset.select(range(train_size, len(dataset)))

    print(f"Train: {len(train_set)} samples")
    print(f"Test:  {len(test_set)} samples")

    # Сохраняем split info
    Path(DATA_DIR).mkdir(exist_ok=True)
    with open(f"{DATA_DIR}/split_info.txt", "w") as f:
        f.write(f"Total: {len(dataset)}\n")
        f.write(f"Train: {len(train_set)} (indices 0–{train_size-1})\n")
        f.write(f"Test:  {len(test_set)} (indices {train_size}–{len(dataset)-1})\n")

    print(f"\nSplit info saved to {DATA_DIR}/split_info.txt")
    print("Done!")
    return train_set, test_set


if __name__ == "__main__":
    train, test = download_ljspeech()
