"""The seam between the move decider and `laya`.

Everything above this module talks to the `Agent` protocol, never to `laya` directly.
That keeps `laya` and `torch` out of the import path of the pure logic in
`decider.py`, and lets the test suite substitute a fake so no test has to load the
842 MB checkpoint.

`laya` is imported inside `LocalLayaAgent.__init__` rather than at module scope, so
that `import chessbot.agent` stays cheap for the pure-logic tests and for the web
layer's error paths.

`LocalLayaAgent` also enforces the invariant the decider relies on: a returned choice
is always one of the options that were offered.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class AgentLoadError(RuntimeError):
    """Raised when the Laya checkpoint cannot be loaded."""


@dataclass(frozen=True)
class ChoiceResult:
    """One multiple-choice answer from the model.

    Attributes:
        choice: The selected option key. Guaranteed to be one of the keys offered.
        probabilities: Probability per offered key, renormalised to sum to 1.
        latency_ms: Wall-clock duration of the underlying `predict` call.
    """

    choice: str
    probabilities: dict[str, float]
    latency_ms: float


@runtime_checkable
class Agent(Protocol):
    """The interface `decider.decide_move` depends on."""

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Ask one multiple-choice question and return the model's answer.

        Args:
            state: The state text describing the position and the current stage.
            question_id: The key the answer comes back under.
            instructions: The instruction text for this stage.
            options: Mapping of option key to the label shown to the model. Must be
                non-empty; its keys are the only legal answers.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.
        """
        ...


class LocalLayaAgent:
    """Runs the fine-tuned Laya checkpoint that lives on this machine.

    One instance is shared by the whole application. `predict` is serialised behind a
    lock because the ASGI threadpool may run two moves at once, and one torch module
    entered from two threads is not a safe assumption.
    """

    def __init__(
        self, model_dir: str | os.PathLike[str], device: str = "cpu", *, load: bool = True
    ) -> None:
        """Load the checkpoint.

        Args:
            model_dir: Path to the fine-tuned checkpoint directory.
            device: Torch device string. `"cpu"` unless a GPU is present.
            load: When `False`, skip loading entirely and leave the wrapper empty. Only
                the tests use it; it is how they substitute a stub without paying for the
                real load.

        Raises:
            AgentLoadError: If `laya` is not importable, or the checkpoint cannot be
                loaded. Callers are expected to catch this and continue in a degraded
                state rather than refuse to boot.
        """
        self.model_dir = os.fspath(model_dir)
        self.device = device
        self._lock = threading.Lock()
        self._laya: Any | None = None

        if not load:
            logger.debug("skipping checkpoint load for %s", self.model_dir)
            return

        started = time.perf_counter()
        logger.info("loading Laya checkpoint from %s on %s", self.model_dir, device)
        try:
            import laya
        except ImportError as exc:
            raise AgentLoadError(f"the laya package is not importable: {exc}") from exc
        try:
            self._laya = laya.Agent(self.model_dir, device=device)
        except Exception as exc:
            raise AgentLoadError(f"could not load {self.model_dir}: {exc}") from exc
        logger.info("Laya checkpoint loaded in %.1fs", time.perf_counter() - started)

    def predict_choice(
        self, state: str, question_id: str, instructions: str, options: dict[str, str]
    ) -> ChoiceResult:
        """Ask one multiple-choice question.

        Args:
            state: The state text describing the position and the current stage.
            question_id: The key the answer comes back under.
            instructions: The instruction text for this stage.
            options: Mapping of option key to label. Every key is a legal answer,
                because callers build it from `board.legal_moves`.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.

        Raises:
            ValueError: If `options` is empty.
            RuntimeError: Any error from `laya.predict` other than the token-budget
                `ValueError`, which is absorbed into a fallback.
        """
        if not options:
            raise ValueError(f"cannot ask {question_id} with no options")

        questions = {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(options),
            }
        }

        started = time.perf_counter()
        try:
            with self._lock:
                raw = self._laya.predict(state, questions)
        except ValueError as exc:
            # Laya raises when a question's option markers do not fit the
            # checkpoint's max_len / head_max_len budget. Failing the whole move is
            # worse than playing the first option, so degrade instead.
            logger.warning(
                "laya rejected question %s (%s); falling back to the first option",
                question_id,
                exc,
            )
            first = next(iter(options))
            return ChoiceResult(
                choice=first,
                probabilities={key: 1.0 if key == first else 0.0 for key in options},
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        return self._coerce_answer(
            raw, question_id, options, (time.perf_counter() - started) * 1000.0
        )

    @staticmethod
    def _coerce_answer(
        raw: dict[str, Any], question_id: str, options: dict[str, str], latency_ms: float
    ) -> ChoiceResult:
        """Normalise laya's answer into a `ChoiceResult` over exactly `options`.

        Three defensive steps happen here. laya rounds probabilities to four decimals,
        so they are renormalised over the offered keys. An all-zero distribution, which
        should not happen, becomes uniform. And a choice the model names that was never
        offered is replaced by the best offered key: the decider filters candidates on
        the choice, so a key that does not exist would empty the candidate set and leave
        the bot unable to move.

        Args:
            raw: The dict returned by `laya.predict`.
            question_id: The key the answer came back under.
            options: The options that were offered.
            latency_ms: Measured duration of the call.

        Returns:
            A `ChoiceResult` whose `choice` is one of `options`.
        """
        answer = raw.get("answers", {}).get(question_id, {})
        raw_probs = dict(answer.get("probabilities") or {})
        choice = str(answer.get("choice", ""))

        probs = {key: max(0.0, float(raw_probs.get(key, 0.0))) for key in options}
        total = sum(probs.values())
        if total <= 0:
            probs = dict.fromkeys(options, 1.0 / len(options))
        else:
            probs = {key: value / total for key, value in probs.items()}

        if choice not in options:
            fallback = max(probs, key=lambda key: (probs[key], key))
            logger.warning(
                "model returned choice %r for %s, which was not offered; using %r",
                choice,
                question_id,
                fallback,
            )
            choice = fallback

        return ChoiceResult(choice=choice, probabilities=probs, latency_ms=latency_ms)
