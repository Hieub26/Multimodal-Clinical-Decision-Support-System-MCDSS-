"""
CLEAR zero-shot scorer (github.com/peterhan91/CLEAR, Apache-2.0; weights from
the Hugging Face repo peterhan91/CLEAR).

The score of a class is cos(image, "<prompt>") - cos(image, "no <prompt>"),
with the authors' chest X-ray preprocessing. It is not a probability; see
image_ensemble.py for how it is combined with DenseNet.

By default nothing is downloaded at run time: the checkpoint must be in the
Hugging Face cache (or CV_CLEAR_CHECKPOINT_PATH) and the DINOv2 source in the
torch hub cache. scripts/fetch_clear.py fetches both once.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch
from PIL import Image

from app.utils.logger import cv_logger


class ClearUnavailableError(RuntimeError):
    """The CLEAR package, checkpoint or DINOv2 source is missing."""


class ClearScorer:
    def __init__(
        self,
        prompts: dict[str, str],
        device: torch.device,
        checkpoint_path: str | Path | None = None,
        allow_download: bool = False,
    ):
        self.prompts = dict(prompts)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.allow_download = allow_download
        self._model = None
        self._preprocess = None
        self._text: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from clear import tokenize
                from clear.hub import load_pretrained
            except ImportError as e:
                raise ClearUnavailableError(
                    "The CLEAR package is not installed (pip install -r requirements-clear.txt)"
                ) from e
            try:
                model, preprocess = load_pretrained(
                    self.checkpoint_path,
                    device=self.device,
                    local_files_only=not self.allow_download,
                )
            except (FileNotFoundError, OSError) as e:
                raise ClearUnavailableError(
                    f"CLEAR weights or DINOv2 source not cached ({e}). "
                    "Run: python scripts/fetch_clear.py"
                ) from e
            model.eval()
            with torch.no_grad():
                for name, prompt in self.prompts.items():
                    feats = model.encode_text(tokenize([prompt, f"no {prompt}"]).to(self.device)).float()
                    self._text[name] = feats / feats.norm(dim=-1, keepdim=True)
            self._preprocess = preprocess
            self._model = model
            cv_logger.info("CLEAR loaded (%d prompts)", len(self.prompts))

    def score(self, image: Image.Image) -> dict[str, float]:
        """Zero-shot score per class for one image."""
        self.ensure_loaded()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out = {}
            for name, text in self._text.items():
                pos, neg = (feat @ text.T)[0].tolist()
                out[name] = pos - neg
        return out
