"""
Corpus Cleaning and Merging
===========================

Merges one language's raw shards (curated downloads + our own crawl) into a
single cleaned corpus, applying:

- NFKC Unicode normalisation
- URL and emoji stripping, whitespace collapsing
- Length and word-count filtering
- Devanagari script-ratio filtering (drops code-mixed text)
- OCR/noise filtering (punctuation ratio, absurd word lengths)
- MinHash-LSH near-deduplication

Both corpora use Devanagari, so the same filters apply to each; the two are
still cleaned in separate runs into separate files and never merged.

Invoked through ``python main.py clean --lang {hindi,nepali}``.
"""

import json
import re
import string
import unicodedata
import multiprocessing as mp

from datasketch import MinHash, MinHashLSH

from src.languages import Language

# Cleaning thresholds
MIN_LENGTH = 100
MIN_DEVANAGARI_RATIO = 0.95
MAX_PUNCTUATION_RATIO = 0.20
MIN_WORD_COUNT = 10
MAX_WORD_LENGTH = 30

# Regex patterns
RE_MULTIPLE_SPACES = re.compile(r'[ \t]+')
RE_MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
RE_DEVANAGARI = re.compile(r'[\u0900-\u097F]')
RE_ALPHA = re.compile(r'[a-zA-Z\u0900-\u097F]')

# Cleaning patterns
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_EMOJI = re.compile(
    r'['
    r'\U0001f600-\U0001f64f'  # emoticons
    r'\U0001f300-\U0001f5ff'  # symbols & pictographs
    r'\U0001f680-\U0001f6ff'  # transport & map symbols
    r'\U0001f1e0-\U0001f1ff'  # flags
    r'\u2600-\u26FF'          # misc symbols
    r'\u2700-\u27BF'          # dingbats
    r']+', flags=re.UNICODE)

def clean_text(text: str) -> str:
    # NFKC Normalization
    text = unicodedata.normalize('NFKC', text)
    # Remove URLs
    text = RE_URL.sub('', text)
    # Remove Emojis
    text = RE_EMOJI.sub('', text)
    # Collapse multiple spaces and newlines
    text = RE_MULTIPLE_SPACES.sub(' ', text)
    text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)
    return text.strip()

def process_chunk(lines: list) -> tuple:
    results = []
    dropped_short = 0
    dropped_ratio = 0
    dropped_noise = 0
    dropped_word_length = 0
    bytes_read = 0
    docs_read = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        docs_read += 1
        bytes_read += len(line.encode('utf-8'))
        
        try:
            record = json.loads(line)
            text = record.get("text", "")
            if not text and isinstance(record, dict) and "content" in record:
                text = record["content"]
        except Exception:
            continue
            
        # 1. Normalize
        text = clean_text(text)
        
        # 2. Length filter
        if len(text) < MIN_LENGTH:
            dropped_short += 1
            continue
            
        words = text.split()
        if len(words) < MIN_WORD_COUNT:
            dropped_short += 1
            continue
            
        if any(len(w) > MAX_WORD_LENGTH for w in words):
            dropped_word_length += 1
            continue
            
        punct_count = sum(1 for c in text if c in string.punctuation)
        if (punct_count / len(text)) > MAX_PUNCTUATION_RATIO:
            dropped_noise += 1
            continue
            
        # 3. Devanagari ratio filter
        alphas = len(RE_ALPHA.findall(text))
        devanagari_count = len(RE_DEVANAGARI.findall(text))
        
        if alphas > 0:
            if (devanagari_count / alphas) < MIN_DEVANAGARI_RATIO:
                dropped_ratio += 1
                continue
        else:
            dropped_ratio += 1
            continue
            
        # 4. MinHash for near-deduplication (shingle size = 3)
        m = MinHash(num_perm=100)
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3]).encode('utf-8')
            m.update(shingle)
        results.append((text, m))
        
    return results, docs_read, bytes_read, dropped_short, dropped_ratio, dropped_noise, dropped_word_length

def process_language(lang: Language) -> dict:
    """Clean and merge every raw shard for one language.

    Args:
        lang: The language to process. Its raw/ directory is read and its
              interim/ file written; no other language is touched.

    Returns:
        A statistics dict describing what was read, written and dropped.
    """
    print(f"\nProcessing {lang.display} ({lang.model_label}) using multiprocessing...")
    manual_dir = lang.raw_dir / "manual"
    curated_dir = lang.raw_dir / "curated"
    out_file = lang.interim_file
    
    lsh = MinHashLSH(threshold=0.55, num_perm=100)
    doc_id_counter = 0
    
    stats = {
        "files_read": 0,
        "docs_read": 0,
        "bytes_read": 0,
        "docs_written": 0,
        "bytes_written": 0,
        "dropped_short": 0,
        "dropped_ratio": 0,
        "dropped_dup": 0,
        "dropped_noise": 0,
        "dropped_word_length": 0
    }
    
    # Get all jsonl files
    files = []
    for d in [manual_dir, curated_dir]:
        if d.exists():
            for f in d.glob("*.jsonl"):
                if not f.name.endswith(".state.json") and not f.name.startswith("."):
                    files.append(f)
                    
    stats["files_read"] = len(files)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use 10 CPU cores for worker processes (leaving 2 for main/OS)
    num_workers = max(1, mp.cpu_count() - 2)
    chunk_size = 8000
    
    with open(out_file, 'w', encoding='utf-8') as f_out:
        with mp.Pool(processes=num_workers) as pool:
            for file_path in files:
                print(f"  Reading: {file_path.name}")
                
                def line_generator():
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        batch = []
                        for line in f_in:
                            batch.append(line)
                            if len(batch) >= chunk_size:
                                yield batch
                                batch = []
                        if batch:
                            yield batch
                
                # Process chunks asynchronously
                for res, docs, b_read, d_short, d_ratio, d_noise, d_wlen in pool.imap(process_chunk, line_generator()):
                    stats["docs_read"] += docs
                    stats["bytes_read"] += b_read
                    stats["dropped_short"] += d_short
                    stats["dropped_ratio"] += d_ratio
                    stats["dropped_noise"] += d_noise
                    stats["dropped_word_length"] += d_wlen
                    
                    for text, m in res:
                        if len(lsh.query(m)) > 0:
                            stats["dropped_dup"] += 1
                            continue
                        
                        doc_id = f"doc_{doc_id_counter}"
                        doc_id_counter += 1
                        lsh.insert(doc_id, m)
                        
                        out_record = {"text": text}
                        out_line = json.dumps(out_record, ensure_ascii=False) + "\n"
                        f_out.write(out_line)
                        
                        stats["docs_written"] += 1
                        stats["bytes_written"] += len(out_line.encode('utf-8'))
                        
    return stats

def run(lang: Language) -> dict:
    """Clean one language's corpus and write a human-readable summary.

    Args:
        lang: The language to clean.

    Returns:
        The statistics dict from :func:`process_language`.
    """
    summary_lines = []
    summary_lines.append("DATA CLEANING AND MERGING PIPELINE SUMMARY")
    summary_lines.append("==========================================")

    stats = process_language(lang)

    summary_lines.append(f"\n[{lang.key.upper()} — {lang.model_label}]")
    summary_lines.append(f"  Files Read: {stats['files_read']}")
    
    mb_read = stats['bytes_read'] / (1024 * 1024)
    mb_written = stats['bytes_written'] / (1024 * 1024)
    
    summary_lines.append(f"  Initial Size: {mb_read:.2f} MB ({stats['docs_read']:,} documents)")
    summary_lines.append(f"  Final Size:   {mb_written:.2f} MB ({stats['docs_written']:,} documents)")
    summary_lines.append(f"  Reduction:    {((1 - (mb_written/mb_read))*100) if mb_read else 0:.1f}% decrease in size")
    
    summary_lines.append("  Documents Dropped:")
    summary_lines.append(f"    - Duplicates (LSH Near):  {stats['dropped_dup']:,}")
    summary_lines.append(f"    - Too Short/Few Words:    {stats['dropped_short']:,}")
    summary_lines.append(f"    - Low Devanagari Ratio:   {stats['dropped_ratio']:,}")
    summary_lines.append(f"    - OCR Noise (Punct):      {stats['dropped_noise']:,}")
    summary_lines.append(f"    - OCR Noise (Word Len):   {stats['dropped_word_length']:,}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    summary_path = lang.interim_file.parent / "cleaning_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(f"\nSummary saved to {summary_path}")
    return stats


def main():
    import argparse
    from src import languages
    parser = argparse.ArgumentParser(description="Clean and merge one language's raw shards into a single corpus.")
    languages.add_language_arg(parser)
    args = parser.parse_args()

    run(languages.get(args.lang))


if __name__ == "__main__":
    main()

