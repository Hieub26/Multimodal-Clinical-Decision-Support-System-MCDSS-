"""Grad-CAM on a DenseNet-121 (random weights: no checkpoint is needed)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from app.core.cv.grad_cam import GradCAM, GradCAMVisualizer


@pytest.fixture(scope="module")
def densenet():
    torch.manual_seed(0)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 3)
    return model.eval()


def _input(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, 3, 224, 224)


def test_map_is_on_the_final_feature_grid_and_normalised(densenet):
    heatmap = GradCAM(densenet).generate(_input(), class_idx=1)

    assert heatmap.shape == (7, 7)
    assert heatmap.min() >= 0
    assert heatmap.max() == pytest.approx(1.0)


def test_map_matches_the_classifier_input(densenet):
    """The weighted feature map sums to the class logit (Grad-CAM on the
    pooled layer, before its ReLU): the map explains this classifier."""
    x = _input()
    features = torch.relu(densenet.features(x))
    logits = densenet(x)
    weights = densenet.classifier.weight[1]
    contribution = (weights[:, None, None] * features[0]).sum(0).mean() + densenet.classifier.bias[1]

    assert contribution.item() == pytest.approx(logits[0, 1].item(), rel=1e-4, abs=1e-4)


def test_repeated_calls_give_the_same_map(densenet):
    cam = GradCAM(densenet)
    x = _input()
    np.testing.assert_allclose(cam.generate(x.clone(), 0), cam.generate(x.clone(), 0), atol=1e-6)


def test_grayscale_upload_is_drawn_in_gray(densenet, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "image_storage_dir", tmp_path)
    shown = []
    import app.core.cv.grad_cam as module
    original_imshow = module.plt.Axes.imshow

    def record(self, data, *args, **kwargs):
        shown.append(np.asarray(data))
        return original_imshow(self, data, *args, **kwargs)

    monkeypatch.setattr(module.plt.Axes, "imshow", record)
    gray = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1)), mode="L")

    path = GradCAMVisualizer(GradCAM(densenet), ["a", "b", "c"]).generate_and_save(_input(), gray, 2)

    assert path
    original = shown[0]
    assert original.ndim == 3 and original.shape[2] == 3  # RGB, not a colour-mapped 2-D array
    assert (original[..., 0] == original[..., 1]).all() and (original[..., 1] == original[..., 2]).all()
