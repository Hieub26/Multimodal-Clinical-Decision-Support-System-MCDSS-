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


def test_map_from_a_kept_feature_map_is_the_same_map(densenet):
    """The prediction's own feature map gives the map a fresh forward pass
    would: no second pass through the model is needed."""
    x = _input()
    cam = GradCAM(densenet)
    with torch.no_grad():
        features = densenet.features(x)

    np.testing.assert_allclose(
        cam.generate_from_features(features, 1), cam.generate(x, 1), atol=1e-6
    )


def test_map_leaves_no_gradient_on_the_model(densenet):
    """Nothing is stored on the shared model, so requests cannot interfere."""
    GradCAM(densenet).generate(_input(), 0)
    assert all(parameter.grad is None for parameter in densenet.parameters())


def _panels(path) -> list[np.ndarray]:
    """The three panels of a saved figure, as RGB arrays."""
    from app.core.cv.grad_cam import _MARGIN, _TITLE_HEIGHT

    figure = np.asarray(Image.open(path).convert("RGB"))
    width = (figure.shape[1] - 4 * _MARGIN) // 3
    return [
        figure[_TITLE_HEIGHT:figure.shape[0] - _MARGIN, left:left + width]
        for left in (_MARGIN + i * (width + _MARGIN) for i in range(3))
    ]


def test_grayscale_upload_is_drawn_in_gray(densenet, tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "image_storage_dir", tmp_path)
    gray = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (64, 1)), mode="L")

    path = GradCAMVisualizer(GradCAM(densenet), ["a", "b", "c"]).generate_and_save(_input(), gray, 2)

    assert path
    original, heatmap, overlay = _panels(path)
    # Gray, not colour-mapped
    assert (original[..., 0] == original[..., 1]).all() and (original[..., 1] == original[..., 2]).all()
    # The heatmap and the overlay are in colour
    assert (heatmap[..., 0] != heatmap[..., 2]).any()
    assert (overlay[..., 0] != overlay[..., 2]).any()


def test_figure_size_does_not_grow_with_the_upload(densenet, tmp_path, monkeypatch):
    """A full-size chest film is drawn at the same size as a small one."""
    from app.config import settings
    from app.core.cv.grad_cam import PANEL_MAX_SIDE

    monkeypatch.setattr(settings, "image_storage_dir", tmp_path)
    film = Image.new("L", (3000, 2400), 128)

    path = GradCAMVisualizer(GradCAM(densenet), ["a", "b", "c"]).generate_and_save(_input(), film, 0)

    width, height = Image.open(path).size
    assert width < 3 * PANEL_MAX_SIDE + 100
    assert height < PANEL_MAX_SIDE


def test_heatmap_colours_run_from_blue_to_red():
    from app.core.cv.grad_cam import _JET

    assert _JET.shape == (256, 3)
    assert _JET[0].argmax() == 2      # coldest: blue
    assert _JET[255].argmax() == 0    # hottest: red
    assert tuple(_JET[128]) == pytest.approx((127, 255, 127), abs=8)  # middle: green
