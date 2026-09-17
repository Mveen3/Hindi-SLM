"""Data ingestion, cleaning, tokenization, dataset streaming, and synthetic reasoning."""

from src.data.dataset import LMDataset, EvalLMDataset, create_dataloader

try:
    from src.data.clean import run as run_cleaning
except ImportError:
    run_cleaning = None

try:
    from src.data.crawler import run as run_crawler
except ImportError:
    run_crawler = None

try:
    from src.data.download import run as download_datasets, download_language
except ImportError:
    download_datasets = download_language = None

try:
    from src.data.hub import download_checkpoint
except ImportError:
    download_checkpoint = None

try:
    from src.data.split import split_corpus
except ImportError:
    split_corpus = None

try:
    from src.data.tokenizer import train_tokenizer, evaluate_tokenizer
except ImportError:
    train_tokenizer = evaluate_tokenizer = None

__all__ = [
    "LMDataset",
    "EvalLMDataset",
    "create_dataloader",
    "run_cleaning",
    "run_crawler",
    "download_datasets",
    "download_checkpoint",
    "split_corpus",
    "train_tokenizer",
    "evaluate_tokenizer",
]
