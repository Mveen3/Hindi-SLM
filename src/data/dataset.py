"""
Streaming JSONL Dataset for Causal Language Model Pretraining
============================================================

Reads JSONL documents (one {"text": "..."} per line) from disk, tokenizes
on-the-fly using the Phase 1 BPE tokenizer, packs consecutive documents into
fixed-length sequences, and yields (input_ids, target_ids) pairs for causal
language modelling.

Design choices:
    - IterableDataset for streaming: each corpus is too large (6-8 GB) to fit
      in memory.  We stream line-by-line and maintain a token buffer.
    - On-the-fly tokenization avoids the need to materialise a separate
      tokenised corpus on disk.
    - Document packing: consecutive documents are concatenated with </s>
      separators and chunked into max_seq_len blocks to maximise GPU
      utilisation (no wasted padding within sequences).
"""

import json
import torch
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tokenizers import Tokenizer


class LMDataset(IterableDataset):
    """
    Streaming language-model dataset that packs JSONL documents into
    fixed-length sequences.

    Each yielded sample is a dict:
        {
            "input_ids": LongTensor of shape (max_seq_len,),
            "targets":   LongTensor of shape (max_seq_len,),
        }
    where targets[i] = input_ids[i+1] (teacher-forced causal LM objective).

    Args:
        data_path:      Path to JSONL file (each line: {"text": "..."}).
        tokenizer_path: Path to HuggingFace tokenizers JSON file.
        max_seq_len:    Maximum sequence length per sample.
        bos_token_id:   Begin-of-sequence token ID.
        eos_token_id:   End-of-sequence token ID.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer_path = str(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Tokenizer is loaded lazily per worker (see __iter__) to avoid
        # serialisation issues with multiprocess DataLoader.
        self._tokenizer = None

    def _get_tokenizer(self) -> Tokenizer:
        """Lazily load the tokenizer (once per worker process)."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        return self._tokenizer

    def _line_iterator(self):
        """
        Yield raw text strings from the JSONL file.

        When using multiple DataLoader workers, each worker handles a
        different shard of the file lines (round-robin by line index)
        to avoid duplicate data.
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with open(self.data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # Round-robin sharding across workers
                if idx % num_workers != worker_id:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    def __iter__(self):
        """
        Yield packed (input_ids, targets) samples by streaming, tokenizing,
        and chunking documents into max_seq_len blocks.

        Packing strategy:
            1. Tokenize each document → list of token IDs.
            2. Append </s> after each document as a separator.
            3. Accumulate tokens into a buffer.
            4. When buffer has ≥ max_seq_len + 1 tokens, extract a chunk:
               input_ids = buffer[:max_seq_len]
               targets   = buffer[1:max_seq_len+1]
            5. Remove the consumed tokens (keeping 1 overlap for the next target).
        """
        tokenizer = self._get_tokenizer()
        buffer = []

        for text in self._line_iterator():
            # Tokenize the document
            encoding = tokenizer.encode(text)
            token_ids = encoding.ids

            if not token_ids:
                continue

            # Append tokens + end-of-sequence separator
            buffer.extend(token_ids)
            buffer.append(self.eos_token_id)

            # Yield as many full sequences as possible
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "targets": targets}
                # Advance buffer — no overlap to avoid data leakage
                buffer = buffer[self.max_seq_len:]

        # Handle remaining tokens in the buffer (if any)
        # Discard partial sequences shorter than max_seq_len + 1 to avoid
        # needing padding logic in the training loop.


class EvalLMDataset(IterableDataset):
    """
    Evaluation dataset that processes documents individually (no packing)
    so that per-document and per-byte statistics can be computed accurately.

    Each yielded sample is a dict:
        {
            "input_ids":  LongTensor of shape (T,),
            "targets":    LongTensor of shape (T,),
            "num_tokens": int — number of non-padding target tokens,
            "num_bytes":  int — byte length of the original text (for BPB).
        }

    Documents longer than max_seq_len are split into non-overlapping chunks.

    Args:
        data_path:      Path to JSONL file.
        tokenizer_path: Path to tokenizer JSON file.
        max_seq_len:    Maximum sequence length.
        pad_token_id:   Padding token ID.
        eos_token_id:   End-of-sequence token ID.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 3,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer_path = str(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._tokenizer = None

    def _get_tokenizer(self) -> Tokenizer:
        """Lazily load the tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = Tokenizer.from_file(self.tokenizer_path)
        return self._tokenizer

    def __iter__(self):
        """Yield individual document chunks with byte-count metadata."""
        tokenizer = self._get_tokenizer()

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if not text:
                        continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                num_bytes = len(text.encode("utf-8"))
                encoding = tokenizer.encode(text)
                token_ids = encoding.ids

                if len(token_ids) < 2:
                    continue

                # Append EOS
                token_ids.append(self.eos_token_id)

                # Split into max_seq_len chunks
                for start in range(0, len(token_ids) - 1, self.max_seq_len):
                    end = min(start + self.max_seq_len + 1, len(token_ids))
                    chunk = token_ids[start:end]

                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    targets = torch.tensor(chunk[1:], dtype=torch.long)
                    seq_len = len(input_ids)

                    # Pad if shorter than max_seq_len
                    if seq_len < self.max_seq_len:
                        pad_len = self.max_seq_len - seq_len
                        input_ids = torch.cat([
                            input_ids,
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                        ])
                        targets = torch.cat([
                            targets,
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                        ])

                    # Byte count is attributed to the first chunk only
                    chunk_bytes = num_bytes if start == 0 else 0

                    yield {
                        "input_ids": input_ids,
                        "targets": targets,
                        "num_tokens": seq_len,
                        "num_bytes": chunk_bytes,
                    }


def create_dataloader(
    data_path: str,
    tokenizer_path: str,
    max_seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 2,
    is_eval: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Args:
        data_path:      Path to JSONL data file.
        tokenizer_path: Path to tokenizer JSON file.
        max_seq_len:    Maximum sequence length.
        batch_size:     Batch size.
        num_workers:    Number of DataLoader workers.
        is_eval:        If True, uses EvalLMDataset (no packing).

    Returns:
        DataLoader instance.
    """
    if is_eval:
        dataset = EvalLMDataset(
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            pad_token_id=kwargs.get("pad_token_id", 0),
            eos_token_id=kwargs.get("eos_token_id", 3),
        )
    else:
        dataset = LMDataset(
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            bos_token_id=kwargs.get("bos_token_id", 2),
            eos_token_id=kwargs.get("eos_token_id", 3),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
