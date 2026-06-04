"""Download LJSpeech from HuggingFace and make a train/test split.

LJSpeech: 13100 text-audio pairs, single (female) voice, ~2.6GB.
"""

from datasets import load_dataset

from core import config
from core.logger import logger

DATA_DIR = str(config.DATA_DIR)


def download_ljspeech():
    logger.info('Downloading LJSpeech from HuggingFace...')
    logger.info('(~2.6GB, this will take a few minutes)')

    dataset = load_dataset(config.DATASET_ID, split='train', cache_dir=DATA_DIR)

    logger.info(f'Total samples: {len(dataset)}')
    logger.info(f'Columns: {dataset.column_names}')
    logger.info(f'Sample text: {dataset[0]["normalized_text"]}')
    logger.info(f'Sample audio: {dataset[0]["audio"]["sampling_rate"]} Hz')

    # Standard split: last TEST_SIZE clips = test
    train_size = len(dataset) - config.TEST_SIZE

    train_set = dataset.select(range(train_size))
    test_set = dataset.select(range(train_size, len(dataset)))

    logger.info(f'Train: {len(train_set)} samples')
    logger.info(f'Test:  {len(test_set)} samples')

    # Save split info
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(f'{DATA_DIR}/split_info.txt', 'w') as f:
        f.write(f'Total: {len(dataset)}\n')
        f.write(f'Train: {len(train_set)} (indices 0-{train_size - 1})\n')
        f.write(f'Test:  {len(test_set)} (indices {train_size}-{len(dataset) - 1})\n')

    logger.info(f'Split info saved to {DATA_DIR}/split_info.txt')
    logger.success('Done!')
    return train_set, test_set


if __name__ == '__main__':
    train, test = download_ljspeech()
