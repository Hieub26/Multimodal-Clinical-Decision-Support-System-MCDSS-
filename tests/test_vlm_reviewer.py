"""The OpenAI image safety reviewer, with a fake client (no API call)."""

import base64
import io

import pytest
from PIL import Image

from app.core.safety.vlm_reviewer import MAX_IMAGE_SIDE, OpenAIVLMReviewer, VLMSafetyReview


class _Response:
    def __init__(self, parsed):
        self.output_parsed = parsed


class _FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def parse(self, **request):
        self.calls.append(request)
        return _Response(self.parsed)


class _FakeClient:
    def __init__(self, parsed):
        self.responses = _FakeResponses(parsed)


VERDICT = VLMSafetyReview(
    decision="caution", risk_level="moderate", image_text_alignment="partially_aligned",
    conflict_resolution="text says pneumonia, image flags effusion",
    safety_context_note="", reasoning="partly aligned", recommended_safety_action="review",
)


@pytest.fixture
def big_xray(tmp_path):
    path = tmp_path / "xray.png"
    Image.new("L", (2048, 1536), 128).save(path)
    return str(path)


def test_request_carries_the_prompt_and_a_downscaled_image(big_xray):
    reviewer = OpenAIVLMReviewer(api_key="test", model="vision-model", reasoning_effort="low")
    reviewer._client = _FakeClient(VERDICT)

    review = reviewer.review("check this case", big_xray)

    request = reviewer._client.responses.calls[0]
    assert request["model"] == "vision-model"
    assert request["text_format"] is VLMSafetyReview
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "low"}
    text, image = request["input"][0]["content"]
    assert text == {"type": "input_text", "text": "check this case"}
    assert image["type"] == "input_image"
    png = base64.b64decode(image["image_url"].split(",", 1)[1])
    assert max(Image.open(io.BytesIO(png)).size) == MAX_IMAGE_SIDE
    assert review["decision"] == "caution" and review["model"] == "vision-model"


def test_no_reasoning_field_for_a_model_without_it(big_xray):
    reviewer = OpenAIVLMReviewer(api_key="test", model="m", reasoning_effort="")
    reviewer._client = _FakeClient(VERDICT)
    reviewer.review("p", big_xray)
    assert "reasoning" not in reviewer._client.responses.calls[0]


def test_a_refusal_raises(big_xray):
    reviewer = OpenAIVLMReviewer(api_key="test", model="m")
    reviewer._client = _FakeClient(None)
    with pytest.raises(ValueError):
        reviewer.review("p", big_xray)


def test_unavailable_without_a_key():
    assert OpenAIVLMReviewer(api_key="").available is False
