"""VITS fine-tuning package.

Importing the package configures the espeak-ng library that phonemizer (the VITS
tokenizer backend) depends on, so submodules can tokenize without per-module setup.
"""

from core.espeak import setup_espeak

setup_espeak()
