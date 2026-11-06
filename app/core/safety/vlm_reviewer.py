"""
Image safety review on an OpenAI vision model.

The reviewer looks at the chest X-ray together with the pipeline's output and
says whether the case should be shown, qualified or escalated. It runs on a
different provider from the model that writes the text diagnosis (Gemini), so
in a text + image case the generator does not review its own answer.

Its verdict can only add caution: SafetyController never lets it clear a risk
(see SafetyController._apply_vlm_review). It has not been evaluated, so it is
not trusted to make a case look safer than the rules already say it is.
"""

import base64
import io
import threading
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

from app.config import settings
from app.utils.logger import safety_logger

# Longest side of the image sent to the model: enough to read a chest film,
# and it bounds the number of image tokens per review.
MAX_IMAGE_SIDE = 1024


class VLMSafetyReview(BaseModel):
    """The reviewer's verdict (structured output)."""
    decision: Literal["approve", "caution", "block", "escalate"]
    risk_level: Literal["low", "moderate", "high", "critical"]
    image_text_alignment: Literal["aligned", "partially_aligned", "conflicting", "unclear"]
    safety_risk_assessed: bool = True
    conflict_resolution: str = Field(description="Consistency or conflicts found, briefly")
    safety_context_note: str = Field(description="Qualified safety statement, if needed")
    reasoning: str = Field(description="Brief safety and escalation reasoning")
    recommended_safety_action: str = Field(description="Precautions or escalation the app should take")


def encode_image(image_path: str) -> str:
    """The image as a PNG data URL, downscaled to MAX_IMAGE_SIDE."""
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenAIVLMReviewer:
    """Asks an OpenAI vision model for a safety verdict on one image case."""

    provider = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
    ):
        self._api_key = (
            api_key if api_key is not None
            else settings.openai_api_key.get_secret_value()
        )
        self.model = model or settings.vlm_safety_model
        self._timeout = timeout or settings.vlm_timeout_seconds
        self._reasoning_effort = (
            settings.vlm_reasoning_effort if reasoning_effort is None
            else reasoning_effort
        )
        self._client = None
        self._client_lock = threading.Lock()
        if self._api_key and not HAS_OPENAI:
            safety_logger.warning(
                "OPENAI_API_KEY is set but the openai package is not "
                "installed: the image safety review is skipped"
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key and HAS_OPENAI)

    def review(self, prompt: str, image_path: str) -> dict:
        """The verdict as a dict. Raises when the call fails."""
        request = {
            "model": self.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": encode_image(image_path)},
                ],
            }],
            "text_format": VLMSafetyReview,
            "store": False,
        }
        if self._reasoning_effort:
            request["reasoning"] = {"effort": self._reasoning_effort}
        response = self._get_client().responses.parse(**request)
        verdict = response.output_parsed
        if verdict is None:
            raise ValueError("the model returned no verdict (refusal or empty output)")
        return {**verdict.model_dump(), "model": self.model}

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = OpenAI(
                        api_key=self._api_key,
                        timeout=self._timeout,
                        # A failure falls back to the rule-based review
                        max_retries=0,
                        # As for the guideline judge: gzip only, see
                        # app/core/validation/guideline_judge.py
                        default_headers={"Accept-Encoding": "gzip"},
                    )
        return self._client
