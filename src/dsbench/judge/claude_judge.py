"""Claude-based judge for scoring answers against a rubric."""

import json
import os

from anthropic import Anthropic


def parse_verdict(text: str) -> dict:
    """Parse judge output (plain or fenced JSON) into score and reasoning."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]

    data = json.loads(text)
    score = float(data["score"])
    score = max(0.0, min(1.0, score))
    return {"score": score, "reasoning": data["reasoning"]}


class ClaudeJudge:
    """Judge that asks a Claude model to score an answer."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kwargs,
    ):
        self.model = model
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key)

    def score(self, prompt: str, answer: str, rubric: str) -> dict:
        """Return a dict with 'score' (0..1) and 'reasoning'."""
        user_message = (
            f"Prompt:\n{prompt}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Rubric:\n{rubric}\n\n"
            "Return only JSON with 'score' (a float between 0 and 1) and "
            "'reasoning' (a string)."
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_message}],
        )

        return parse_verdict(response.content[0].text)
