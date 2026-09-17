"""
HuggingFace Hub Checkpoint Transfer
===================================

Pulls a trained checkpoint down from the Hub into ``<language>/checkpoints/``
or ``<language>/checkpoints_finetune/``, or pushes a local checkpoint up to
the Hub.

Checkpoints are stored in the unified repository ``mveen3/hindi_slm`` under
dedicated subfolders:
    - ``<language>/pretraining/``
    - ``<language>/finetuning/``

Requires ``HF_TOKEN`` in the project-root ``.env`` (or the environment).

Invoked through ``python main.py hub --lang {hindi,nepali} [--stage {pretraining,finetuning}] [--upload]``.
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


def find_latest_step(lang: Language, repo_id: str, stage: str = "pretraining") -> tuple[int, str]:
    """Return the highest checkpoint step and remote path available for this language.

    Args:
        lang:    Language whose checkpoints are listed.
        repo_id: HuggingFace repo to search.
        stage:   "pretraining" or "finetuning".

    Returns:
        Tuple of (highest step number, remote file path).

    Raises:
        SystemExit: If the repo holds no checkpoint for this language and stage.
    """
    subfolder = f"{lang.key}/{stage}"
    pattern = re.compile(rf"{subfolder}/checkpoint_step_(\d+)\.pt")
    files = list_repo_files(repo_id)

    steps = [
        (int(m.group(1)), f)
        for f in files
        if (m := pattern.fullmatch(f))
    ]

    # Fallback to alternative spelling (e.g. pretraing / fintuning) or root
    if not steps:
        alt_stage = "pretraing" if stage == "pretraining" else "fintuning"
        alt_pattern = re.compile(rf"{lang.key}/(?:{alt_stage}/)?checkpoint_step_(\d+)\.pt")
        steps = [
            (int(m.group(1)), f)
            for f in files
            if (m := alt_pattern.fullmatch(f))
        ]

    if not steps:
        raise SystemExit(f"No {lang.key} {stage} checkpoints found in {repo_id} ({subfolder}/).")
    return max(steps, key=lambda x: x[0])


def download_checkpoint(lang: Language, step: int = None, repo_id: str = None, stage: str = "pretraining"):
    """
    Fetch one checkpoint into the language's own checkpoints directory.

    Args:
        lang:    Language to fetch for.
        step:    Checkpoint step; defaults to the latest available.
        repo_id: Override for the language's default Hub repo.
        stage:   "pretraining" or "finetuning".

    Returns:
        Path to the local checkpoint file.
    """
    repo_id = repo_id or lang.hf_repo
    login(token=resolve_token())

    if step is None:
        step, filename = find_latest_step(lang, repo_id, stage=stage)
    else:
        filename = f"{lang.key}/{stage}/checkpoint_step_{step}.pt"

    dest_dir = lang.checkpoint_dir if stage == "pretraining" else lang.finetune_checkpoint_dir
    dest = dest_dir / Path(filename).name

    if dest.exists():
        logger.info("Already present, nothing to do: %s", dest)
        return dest

    logger.info("Downloading %s from %s ...", filename, repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)

    shutil.copyfile(cached, dest)
    logger.info("Saved to %s", dest)
    return dest


def upload_checkpoint(lang: Language, checkpoint_path: str, repo_id: str = None, stage: str = "pretraining"):
    """
    Upload a checkpoint to <lang>/<stage>/ in the HuggingFace Hub repository.

    Args:
        lang:            Language whose checkpoint is uploaded.
        checkpoint_path: Path to local .pt file.
        repo_id:         Target repository.
        stage:           "pretraining" or "finetuning".
    """
    repo_id = repo_id or lang.hf_repo
    login(token=resolve_token())
    api = HfApi()

    path = Path(checkpoint_path)
    if not path.exists():
        raise SystemExit(f"Checkpoint file not found: {path}")

    subfolder = f"{lang.key}/{stage}"
    # Ensure subfolder exists
    try:
        files = api.list_repo_files(repo_id=repo_id)
        if not any(f.startswith(f"{subfolder}/") for f in files):
            api.upload_file(
                path_or_fileobj=b"",
                path_in_repo=f"{subfolder}/.gitkeep",
                repo_id=repo_id,
                commit_message=f"Initialize {subfolder} subfolder",
            )
    except Exception as exc:
        logger.debug("Subfolder check note: %s", exc)

    logger.info("Uploading %s to %s/%s ...", path.name, repo_id, subfolder)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"{subfolder}/{path.name}",
        repo_id=repo_id,
        commit_message=f"Upload {lang.display} {stage} checkpoint ({path.name})",
    )
    logger.info("Uploaded successfully to https://huggingface.co/%s/tree/main/%s", repo_id, subfolder)


def main():
    import argparse
    from src import languages
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Pull or push trained checkpoints from/to HuggingFace Hub.")
    languages.add_language_arg(parser)
    parser.add_argument("--step", type=int, default=None,
                        help="Checkpoint step (default: the latest available).")
    parser.add_argument("--repo-id", default=None,
                        help="Override the language's default Hub repo (default: mveen3/hindi_slm).")
    parser.add_argument("--stage", choices=["pretraining", "finetuning"], default="pretraining",
                        help="Target subfolder stage: pretraining or finetuning (default: pretraining).")
    parser.add_argument("--upload", action="store_true", default=False,
                        help="Upload a local checkpoint instead of downloading.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (used with --upload; defaults to latest local checkpoint).")
    args = parser.parse_args()

    lang = languages.get(args.lang)
    if args.upload:
        ckpt_path = args.checkpoint
        if not ckpt_path:
            from src.train.utils import find_latest_checkpoint
            search_dir = str(lang.checkpoint_dir if args.stage == "pretraining" else lang.finetune_checkpoint_dir)
            ckpt_path = find_latest_checkpoint(search_dir)
            if not ckpt_path:
                raise SystemExit(f"No local {args.stage} checkpoints found in {search_dir} to upload.")
        upload_checkpoint(lang, ckpt_path, repo_id=args.repo_id, stage=args.stage)
    else:
        download_checkpoint(lang, step=args.step, repo_id=args.repo_id, stage=args.stage)


if __name__ == "__main__":
    main()

