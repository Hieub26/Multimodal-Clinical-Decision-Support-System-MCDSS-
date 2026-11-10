"""
Fetch what CLEAR needs at run time, once, so the app never downloads it:
the checkpoint (Hugging Face peterhan91/CLEAR, about 1 GB) and the DINOv2
source code (torch hub facebookresearch/dinov2, needed to build the backbone;
its pretrained weights are not used, the CLEAR checkpoint carries them).

    pip install -r requirements-clear.txt
    python scripts/fetch_clear.py

Both are third-party downloads: check the CLEAR and DINOv2 licenses and the
data-use terms of the datasets CLEAR was trained on before using it clinically.
"""

from clear.hub import load_pretrained

if __name__ == "__main__":
    load_pretrained(device="cpu", local_files_only=False)
    print("CLEAR checkpoint and DINOv2 source are cached")
