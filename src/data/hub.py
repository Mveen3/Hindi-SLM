"""
HuggingFace Hub Checkpoint Transfer
===================================

Pulls a trained checkpoint down from the Hub into ``<language>/checkpoints/``.

Checkpoints are ~280 MB each, so they are not committed to git — the brief asks
for large artifacts to live off-repo with links in the README. Training on
Kaggle pushes them up (see :mod:`src.train.kaggle`); this module pulls them back
down for local inference and evaluation.

Each language has its own repo, named in :mod:`src.languages`, so the two
models' weights are never co-located.

Requires ``HF_TOKEN`` in the project-root ``.env`` (or the environment).

Invoked through ``python main.py hub --lang {hindi,nepali}``.
"""

import os
import re
import shutil
import logging

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files, login

from src.languages import Language, PROJECT_ROOT

logger = logging.getLogger(__name__)


def resolve_token() -> str:
    """Read HF_TOKEN from the project .env, falling back to the environment.

    Returns:
        The token string.

    Raises:
        SystemExit: If no token is configured.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            f"HF_TOKEN not found. Add it to {env_path} or export it in your shell."
        )
    return token


def find_latest_step(lang: Language, repo_id: str) -> int:
    """Return the highest checkpoint step available for this language.

    Args:
        lang:    Language whose checkpoints are listed.
        repo_id: HuggingFace repo to search.

    Returns:
        The highest step number found.

    Raises:
        SystemExit: If the repo holds no checkpoint for this language.
    """
    pattern = re.compile(rf"{lang.key}/checkpoint_step_(\d+)\.pt")
    steps = [
        int(m.group(1))
        for f in list_repo_files(repo_id)
        if (m := pattern.fullmatch(f))
    ]
    if not steps:
        raise SystemExit(f"No {lang.key} checkpoints found in {repo_id}.")
    return max(steps)


def download_checkpoint(lang: Language, step: int = None, repo_id: str = None):
    """
    Fetch one checkpoint into the language's own checkpoints/ directory.

    Args:
        lang:    Language to fetch for.
        step:    Checkpoint step; defaults to the latest available.
        repo_id: Override for the language's default Hub repo.

    Returns:
        Path to the local checkpoint file.
    """
    repo_id = repo_id or lang.hf_repo
    login(token=resolve_token())

    step = step if step is not None else find_latest_step(lang, repo_id)
    filename = f"{lang.key}/checkpoint_step_{step}.pt"
    dest = lang.checkpoint_dir / f"checkpoint_step_{step}.pt"

    if dest.exists():
        logger.info("Already present, nothing to do: %s", dest)
        return dest

    logger.info("Downloading %s from %s ...", filename, repo_id)
    lang.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)

    # hf_hub_download returns a path inside the shared HF cache; copy it into
    # the language's own directory so the layout stays self-contained.
    shutil.copyfile(cached, dest)
    logger.info("Saved to %s", dest)
    return dest


def main():
    import argparse
    from src import languages
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Pull a trained checkpoint from the HuggingFace Hub.")
    languages.add_language_arg(parser)
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: the latest available).")
    parser.add_argument("--repo-id", default=None,
                        help="Override the language's default Hub repo.")
    args = parser.parse_args()

    download_checkpoint(languages.get(args.lang), step=args.step, repo_id=args.repo_id)


if __name__ == "__main__":
    main()

