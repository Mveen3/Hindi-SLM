"""
Language Registry
=================

The single place that knows anything language-specific.

Model H (Hindi) and Model L (Nepali) share this implementation package but
share *nothing* at runtime: each has its own corpus, tokenizer, vocabulary,
configuration and weights, all of which live under ``<project>/<language>/``.
This module resolves those per-language paths so that no other module in
``src`` needs to hard-code a language name.

Adding a third language means adding one entry here plus a
``<language>/configs/`` directory — no other file changes.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Language:
    """Static metadata describing one of the two independent models.

    Attributes:
        key:           Directory name and CLI value, e.g. ``"hindi"``.
        display:       Human-readable name used in plot titles and logs.
        model_label:   ``"Model H"`` (higher-resource) or ``"Model L"`` (lower-resource).
        tier:          Resource tier, for report text.
        hf_repo:       HuggingFace Hub repo holding this model's checkpoints.
        tokenizer_probe: Short in-language string used as a tokenizer sanity check.
        attention_samples: In-language sentences used for attention heatmaps.
    """

    key: str
    display: str
    model_label: str
    tier: str
    hf_repo: str
    tokenizer_probe: str
    attention_samples: tuple = field(default=())

    # ----- Directory layout (all relative to <project>/<key>/) -------------

    @property
    def root(self) -> Path:
        """The language's self-contained artifact directory."""
        return PROJECT_ROOT / self.key

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def tokenizer_path(self) -> Path:
        """Trained BPE tokenizer for this language (never shared)."""
        return self.root / "tokenizer" / "tokenizer.json"

    @property
    def vocab_path(self) -> Path:
        return self.root / "tokenizer" / "vocabulary.txt"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Pretraining loss histories, kept in logs_pretraining."""
        return self.root / "logs_pretraining"

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim_file(self) -> Path:
        return self.root / "data" / "interim" / f"{self.key}_cleaned.jsonl"

    @property
    def report_dir(self) -> Path:
        """Where this language's graded figures and tables are written."""
        return PROJECT_ROOT / "report" / self.key

    # ----- Phase 3: reasoning finetuning artifacts ------------------------

    @property
    def finetune_checkpoint_dir(self) -> Path:
        """Finetuned checkpoints, kept apart from the pretrained ones.

        Separate from ``checkpoint_dir`` on purpose: the Phase 2 checkpoint
        must survive intact as the "pretrained" arm of the Phase 3 comparison.
        """
        return self.root / "checkpoints_finetune"

    @property
    def finetune_log_dir(self) -> Path:
        """Finetuning loss histories, kept apart from the pretraining logs."""
        return self.root / "logs_finetune"

    # ----- Phase 3: synthetic reasoning finetuning data -------------------

    @property
    def reasoning_dir(self) -> Path:
        """Where this language's synthetic reasoning corpus lives (never shared)."""
        return self.root / "data" / "reasoning"

    @property
    def reasoning_phrasebank(self) -> Path:
        """Cached in-language phrasing bank used to render reasoning examples."""
        return self.reasoning_dir / "phrasebank.json"

    def reasoning_split_file(self, split: str) -> Path:
        """Path to a reasoning finetuning split (``train`` / ``val`` / ``test``)."""
        return self.reasoning_dir / f"{split}.jsonl"

    def split_file(self, split: str) -> Path:
        """Path to a processed split file (``train`` / ``val`` / ``test``)."""
        return self.root / "data" / "processed" / f"{split}.jsonl"

    def title(self, what: str) -> str:
        """Build a consistent figure/log title, e.g. 'Loss — Hindi (Model H)'."""
        return f"{what} — {self.display} ({self.model_label})"

    # ----- Configuration ---------------------------------------------------

    def load_config(self, model_config: str = None, train_config: str = None) -> dict:
        """Load and merge this language's model + training YAML configs.

        Args:
            model_config: Override path to model_config.yaml.
            train_config: Override path to train_config.yaml.

        Returns:
            A single merged configuration dictionary. Model keys come first so
            a training config can never silently shadow an architecture key.
        """
        model_path = Path(model_config) if model_config else self.config_dir / "model_config.yaml"
        train_path = Path(train_config) if train_config else self.config_dir / "train_config.yaml"

        with open(model_path) as f:
            model_cfg = yaml.safe_load(f)
        with open(train_path) as f:
            train_cfg = yaml.safe_load(f)

        config = {**model_cfg, **train_cfg}
        config["language"] = self.key
        return config

    def load_finetune_config(self, path: str = None) -> dict:
        """Load this language's model config merged with its finetuning config.

        The model architecture must be identical to pretraining (the brief
        requires the tokenizer and vocabulary to stay fixed), so the
        architecture keys come from ``model_config.yaml`` and only the
        optimisation keys come from ``finetune_config.yaml``.

        Args:
            path: Override path to ``finetune_config.yaml``.

        Returns:
            The merged configuration dictionary.

        Raises:
            SystemExit: If the finetuning config is missing.
        """
        config_path = Path(path) if path else self.config_dir / "finetune_config.yaml"
        if not config_path.exists():
            raise SystemExit(
                f"{config_path} not found — every language needs its own "
                f"finetune_config.yaml for Phase 3."
            )
        with open(self.config_dir / "model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)
        with open(config_path) as f:
            finetune_cfg = yaml.safe_load(f) or {}

        config = {**model_cfg, **finetune_cfg}
        config["language"] = self.key
        return config

    def load_reasoning_config(self, path: str = None) -> dict:
        """Load this language's Phase 3 reasoning-data generation config.

        Args:
            path: Override path to ``reasoning_config.yaml``.

        Returns:
            The parsed configuration dictionary.

        Raises:
            SystemExit: If the config file is missing.
        """
        config_path = Path(path) if path else self.config_dir / "reasoning_config.yaml"
        if not config_path.exists():
            raise SystemExit(
                f"{config_path} not found — every language needs its own "
                f"reasoning_config.yaml for Phase 3."
            )
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def load_sources(self) -> dict:
        """Load this language's data-source manifest (HF datasets + crawl seeds)."""
        with open(self.config_dir / "data_sources.yaml") as f:
            return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# The two models
# ---------------------------------------------------------------------------

LANGUAGES = {
    "hindi": Language(
        key="hindi",
        display="Hindi",
        model_label="Model H",
        tier="higher-resource",
        hf_repo="mveen3/src-hindi-checkpoints",
        tokenizer_probe="नमस्ते दुनिया",
        attention_samples=(
            "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ बोली जाती हैं",
            "हिन्दी भाषा का विकास एक लंबी प्रक्रिया रही है",
            "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने बहुत प्रगति की है",
        ),
    ),
    "nepali": Language(
        key="nepali",
        display="Nepali",
        model_label="Model L",
        tier="lower-resource",
        hf_repo="mveen3/src-nepali-checkpoints",
        tokenizer_probe="नमस्कार संसार",
        attention_samples=(
            "नेपाल एक सुन्दर हिमाली देश हो जहाँ विभिन्न जातजाति बसोबास गर्छन्",
            "नेपाली भाषाको विकास लामो समयदेखि भइरहेको छ",
            "विज्ञान र प्रविधिको क्षेत्रमा नेपालले उल्लेखनीय प्रगति गरेको छ",
        ),
    ),
}

CHOICES = tuple(LANGUAGES)


def get(key: str) -> Language:
    """Look up a Language by key, with a helpful error for typos."""
    try:
        return LANGUAGES[key.lower()]
    except KeyError:
        raise SystemExit(
            f"Unknown language {key!r}. Choose one of: {', '.join(CHOICES)}"
        )


def add_language_arg(parser, required: bool = True):
    """Attach the standard ``--lang`` option to an argparse parser."""
    parser.add_argument(
        "--lang", required=required, choices=CHOICES,
        help="Which model to operate on: hindi (Model H) or nepali (Model L).",
    )
    return parser
