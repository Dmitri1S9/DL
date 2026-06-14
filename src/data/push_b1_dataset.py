"""Push audio/ljspeech_b1 (wavs + metadata.csv) to the Hub as a Dataset.

13100 individual files would hit HF's per-hour commit rate limit (hf upload
tries to commit ~20 files at a time -> hundreds of commits). push_to_hub()
instead packs everything into a handful of Parquet shards (~500MB each),
which is a small number of commits.

We build the Dataset directly with Dataset.from_dict + cast_column instead of
load_dataset("audiofolder", ...): the audiofolder CSV path hits a pyarrow
large_string vs string mismatch in this datasets/pyarrow combo.

Usage:
    PYTHONPATH=src python -m data.push_b1_dataset
"""

import csv

from datasets import Audio, Dataset, Features, Value

from core import config
from core.logger import logger

DATA_DIR = config.AUDIO_DIR / 'ljspeech_b1'
REPO_ID = 'Dmi1tr13/ljspeech-b1'


def main() -> None:
    meta_path = DATA_DIR / 'metadata.csv'
    with open(meta_path, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    paths = [str(DATA_DIR / row['file_name']) for row in rows]
    texts = [row['text'] for row in rows]
    logger.info(f'{len(paths)} examples from {meta_path}')

    features = Features({'audio': Value('string'), 'text': Value('string')})
    ds = Dataset.from_dict({'audio': paths, 'text': texts}, features=features)
    ds = ds.cast_column('audio', Audio(sampling_rate=22050))

    logger.info(f'Pushing to {REPO_ID}...')
    ds.push_to_hub(REPO_ID)
    logger.success(f'Pushed -> https://huggingface.co/datasets/{REPO_ID}')


if __name__ == '__main__':
    main()
