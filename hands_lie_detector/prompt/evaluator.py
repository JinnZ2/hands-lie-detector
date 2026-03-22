"""
Prompt-based hand experience evaluator.

Sends hand images to a vision-capable LLM with the scoring rubric
as a structured prompt, then parses the response into scores.
"""

import base64
import json
import re
from pathlib import Path

from ..scoring.rubric import ScoringRubric
from ..scoring.evaluator import ScoreEvaluator, ContextModifiers, EvaluationResult


SYSTEM_PROMPT = """You are a hand experience scoring system. You analyze images of hands
to estimate work experience probability using persistent physical markers.

CRITICAL RULES:
- Do NOT equate clean with unused
- Do NOT assume dirt = competence
- Do NOT score based on props, context, or clothing
- DO score post-washing state (structural adaptation over surface appearance)
- DO consider climate context
- DO distinguish maintenance from lack of use

The test: Can you see tissue memory through soap?"""


def _build_scoring_prompt(rubric: ScoringRubric) -> str:
    """Build the scoring instruction prompt from the rubric."""
    lines = [
        "Analyze this hand image and score each category.",
        "Respond ONLY with valid JSON in this exact format:",
        "{",
    ]

    for cat in rubric.categories:
        tier_desc = " | ".join(
            f"{t.min_score}-{t.max_score}: {t.description}" for t in cat.tiers
        )
        lines.append(f'  "{cat.name}": {{"score": <0-{cat.max_points}>, "reasoning": "<brief>"}},')
        lines.append(f"    // {cat.question} ({tier_desc})")

    lines.append('  "cold_climate": <true/false>,')
    lines.append('  "glove_excuse_defensive": <true/false>,')
    lines.append('  "estimated_age_decade": <2-8 or null>')
    lines.append("}")
    lines.append("")
    lines.append("Score based on STRUCTURAL ADAPTATION, not surface appearance.")

    return "\n".join(lines)


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Read and base64-encode an image. Returns (base64_data, media_type)."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_types.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
    return data, media_type


def _parse_response(text: str, rubric: ScoringRubric) -> tuple[dict[str, float], ContextModifiers]:
    """Extract scores and modifiers from LLM JSON response."""
    # Find JSON block in response
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not find JSON in response: {text[:200]}")

    data = json.loads(json_match.group())

    scores = {}
    for cat in rubric.categories:
        entry = data.get(cat.name, {})
        if isinstance(entry, dict):
            scores[cat.name] = float(entry.get("score", 0))
        else:
            scores[cat.name] = float(entry)

    modifiers = ContextModifiers(
        cold_climate=bool(data.get("cold_climate", False)),
        glove_excuse_defensive=bool(data.get("glove_excuse_defensive", False)),
        age_decade=data.get("estimated_age_decade"),
    )

    return scores, modifiers


class PromptEvaluator:
    """
    Evaluate hand images using a vision-capable LLM.

    Supports Anthropic (Claude) and OpenAI (GPT-4o) APIs.

    Args:
        provider: "anthropic" or "openai".
        api_key: API key. If None, reads from ANTHROPIC_API_KEY or OPENAI_API_KEY env var.
        model: Model name override.
        rubric: Custom ScoringRubric (defaults to v0.1).

    Usage:
        evaluator = PromptEvaluator(provider="anthropic")
        result = evaluator.evaluate("path/to/hand.jpg")
        print(result.band_label)  # "Working Hands (Light-Moderate)"
    """

    DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        rubric: ScoringRubric | None = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODELS.get(provider, "")
        self.rubric = rubric or ScoringRubric()
        self.score_evaluator = ScoreEvaluator(self.rubric)
        self._scoring_prompt = _build_scoring_prompt(self.rubric)

    def _get_api_key(self) -> str:
        """Resolve API key from init or environment."""
        if self.api_key:
            return self.api_key
        import os
        env_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        key = os.environ.get(env_var, "")
        if not key:
            raise ValueError(
                f"No API key provided. Set {env_var} or pass api_key to constructor."
            )
        return key

    def _call_anthropic(self, image_path: Path) -> str:
        """Call Anthropic Messages API with image."""
        import anthropic

        client = anthropic.Anthropic(api_key=self._get_api_key())
        image_data, media_type = _encode_image(image_path)

        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": self._scoring_prompt,
                        },
                    ],
                }
            ],
        )
        return message.content[0].text

    def _call_openai(self, image_path: Path) -> str:
        """Call OpenAI Chat API with image."""
        from openai import OpenAI

        client = OpenAI(api_key=self._get_api_key())
        image_data, media_type = _encode_image(image_path)

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                        {
                            "type": "text",
                            "text": self._scoring_prompt,
                        },
                    ],
                },
            ],
        )
        return response.choices[0].message.content

    def evaluate(self, image_path: str | Path) -> EvaluationResult:
        """
        Score a hand image using the configured vision LLM.

        Args:
            image_path: Path to hand image (jpg, png, webp).

        Returns:
            EvaluationResult with per-category scores, modifiers, and band label.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.provider == "anthropic":
            response_text = self._call_anthropic(image_path)
        elif self.provider == "openai":
            response_text = self._call_openai(image_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        scores, modifiers = _parse_response(response_text, self.rubric)
        return self.score_evaluator.evaluate(scores, modifiers)

    def evaluate_batch(
        self, image_paths: list[str | Path]
    ) -> list[EvaluationResult]:
        """Evaluate multiple images sequentially."""
        return [self.evaluate(p) for p in image_paths]
