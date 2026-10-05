"""Test doubles shared across the suite. Nothing here imports `laya` or `torch`."""

from __future__ import annotations

from dataclasses import dataclass, field

from chessbot.agent import ChoiceResult


@dataclass
class FakeAgent:
    """Scripted stand-in for `LocalLayaAgent`.

    Records every question asked, so tests can assert both the number of model calls
    and the exact prompt text.

    Attributes:
        script: Maps a question id to the option key that should win. A value may also be
            a list of keys, consumed one per call, which is how a test scripts a whole
            line such as fool's mate. A question id that is absent, exhausted, or
            present but naming an option that was not offered, falls back to the first
            offered option.
        probabilities: Maps a question id to an explicit probability dict, overriding
            `script` for that question. Use it to put all mass on one option.
        calls: Every `(question_id, state, instructions, options)` asked so far.
    """

    script: dict[str, str | list[str]] = field(default_factory=dict)
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    calls: list[tuple[str, str, str, dict[str, str]]] = field(default_factory=list)

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Return the scripted answer and record the question.

        Args:
            state: The state text.
            question_id: The question key.
            instructions: The instruction text.
            options: The options offered.

        Returns:
            A `ChoiceResult` whose `choice` is always one of `options`, matching the
            guarantee `LocalLayaAgent` provides.
        """
        self.calls.append((question_id, state, instructions, dict(options)))

        if question_id in self.probabilities:
            raw = self.probabilities[question_id]
            chosen = max(options, key=lambda key: raw.get(key, 0.0))
            probs = {key: max(0.0, float(raw.get(key, 0.0))) for key in options}
        else:
            chosen = self._next_choice(question_id, options)
            probs = _peaked(options, chosen)

        total = sum(probs.values()) or 1.0
        return ChoiceResult(
            choice=chosen,
            probabilities={key: value / total for key, value in probs.items()},
            latency_ms=1.0,
        )

    def _next_choice(self, question_id: str, options: dict[str, str]) -> str:
        """Take the next scripted option for a question, falling back to the first.

        Args:
            question_id: The question key.
            options: The options offered.

        Returns:
            One key of `options`.
        """
        scripted = self.script.get(question_id)
        if isinstance(scripted, list):
            while scripted:
                candidate = scripted.pop(0)
                if candidate in options:
                    return candidate
        elif scripted in options:
            return str(scripted)
        return next(iter(options))

    @property
    def question_ids(self) -> list[str]:
        """The question ids asked so far, in order."""
        return [call[0] for call in self.calls]

    @property
    def call_count(self) -> int:
        """How many questions have been asked."""
        return len(self.calls)


def _peaked(options: dict[str, str], winner: str) -> dict[str, float]:
    """Return probabilities giving `winner` most of the mass.

    Args:
        options: The options offered.
        winner: The key that should be most likely.

    Returns:
        A probability per offered key, summing to 1.
    """
    losers = len(options) - 1
    if not losers:
        return {winner: 1.0}
    return {key: (0.7 if key == winner else 0.3 / losers) for key in options}
