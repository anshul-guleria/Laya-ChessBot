"""The agent wrapper's non-model behaviour: normalisation, fallback, serialisation.

No test here loads the checkpoint. `LocalLayaAgent` is exercised by replacing its
internal laya object after construction, which is the seam that keeps the rest of the
suite fast.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chessbot.agent import Agent, ChoiceResult, LocalLayaAgent
from tests.support import FakeAgent

BACKEND_ROOT = Path(__file__).resolve().parents[1]

STUB_RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "piece_selection": {
            "type": "choice",
            "choice": "N",
            "probabilities": {"P": 0.4, "N": 0.6},
        }
    },
    "usage": {},
}


class StubLaya:
    """Stand-in for `laya.Agent`, recording the calls it receives."""

    def __init__(self, result=None, raises=None, delay=0.0):
        self.result = STUB_RESULT if result is None else result
        self.raises = raises
        self.delay = delay
        self.calls: list[tuple[str, dict]] = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        return self.result


def make_agent(stub: StubLaya) -> LocalLayaAgent:
    """Build an agent with a stub in place of the loaded checkpoint.

    `load=False` is what keeps this test suite fast: the constructor skips the 842 MB load
    and the internal object is swapped for a stub.

    Args:
        stub: The stand-in to install.

    Returns:
        A `LocalLayaAgent` whose `predict` goes to `stub`.
    """
    agent = LocalLayaAgent(model_dir="unused", device="cpu", load=False)
    agent._laya = stub
    return agent


def test_predict_choice_returns_the_choice_and_probabilities():
    agent = make_agent(StubLaya())
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"}
    )
    assert isinstance(result, ChoiceResult)
    assert result.choice == "N"
    assert result.latency_ms > 0


def test_predict_choice_sends_the_expected_payload():
    stub = StubLaya()
    agent = make_agent(stub)
    agent.predict_choice("state text", "piece_selection", "Choose.", {"P": "Pawn"})
    assert stub.calls == [
        (
            "state text",
            {
                "piece_selection": {
                    "type": "choice",
                    "instructions": "Choose.",
                    "criteria": {"P": "Pawn"},
                }
            },
        )
    ]


def test_missing_probability_keys_are_filled_with_zero():
    agent = make_agent(StubLaya())
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight", "Q": "Queen"}
    )
    assert set(result.probabilities) == {"P", "N", "Q"}
    assert result.probabilities["Q"] == 0.0


def test_probabilities_are_renormalised_over_the_offered_options():
    """laya rounds to four decimals, so the values rarely sum to exactly 1."""
    stub = StubLaya(
        result={
            "answers": {
                "piece_selection": {"choice": "P", "probabilities": {"P": 0.3333, "N": 0.3333}}
            }
        }
    )
    agent = make_agent(stub)
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"}
    )
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_all_zero_probabilities_fall_back_to_uniform():
    stub = StubLaya(
        result={
            "answers": {"piece_selection": {"choice": "P", "probabilities": {"P": 0.0, "N": 0.0}}}
        }
    )
    agent = make_agent(stub)
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"}
    )
    assert result.probabilities == {"P": 0.5, "N": 0.5}


def test_a_choice_the_model_invents_is_replaced_by_the_best_offered_option():
    agent = make_agent(
        StubLaya(
            result={
                "answers": {
                    "piece_selection": {"choice": "Z", "probabilities": {"P": 0.3, "N": 0.7}}
                }
            }
        )
    )
    result = agent.predict_choice(
        "state", "piece_selection", "Choose.", {"P": "Pawn", "N": "Knight"}
    )
    assert result.choice == "N"
    assert result.choice in {"P", "N"}


def test_token_budget_overflow_degrades_to_the_first_option():
    """Laya raises when a question's options overflow its token budget.

    Losing the move is worse than playing a fallback, so the wrapper absorbs it.
    """
    agent = make_agent(StubLaya(raises=ValueError("only 3 of its 40 option markers fit")))
    result = agent.predict_choice(
        "state", "destination_selection", "Choose.", {"e4": "e4", "d4": "d4"}
    )
    assert result.choice == "e4"
    assert result.probabilities == {"e4": 1.0, "d4": 0.0}


def test_other_model_errors_propagate():
    """Only the token-budget ValueError is absorbed; real failures must surface."""
    agent = make_agent(StubLaya(raises=RuntimeError("out of memory")))
    with pytest.raises(RuntimeError, match="out of memory"):
        agent.predict_choice("state", "piece_selection", "Choose.", {"P": "Pawn"})


def test_predict_rejects_an_empty_option_set():
    agent = make_agent(StubLaya())
    with pytest.raises(ValueError, match="no options"):
        agent.predict_choice("state", "piece_selection", "Choose.", {})


def test_predict_is_serialised_so_two_moves_never_overlap():
    """The ASGI threadpool can run two moves at once; one torch module must not."""
    overlap_sizes: list[int] = []
    state = {"active": 0}

    class OverlapStub(StubLaya):
        def predict(self, state_text, questions):
            state["active"] += 1
            overlap_sizes.append(state["active"])
            time.sleep(0.02)
            state["active"] -= 1
            return STUB_RESULT

    agent = make_agent(OverlapStub())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda _: agent.predict_choice("s", "piece_selection", "c", {"P": "Pawn"}),
                range(4),
            )
        )

    assert max(overlap_sizes) == 1, f"predict calls overlapped: {overlap_sizes}"


def test_importing_the_module_does_not_pull_in_torch_or_laya():
    """`import laya` must stay inside `__init__`, or every pure test pays its import cost.

    Checked in a fresh interpreter so that a module another test already imported cannot
    mask a module-scope `import laya` here.
    """
    probe = "import chessbot.agent, sys; assert 'laya' not in sys.modules, 'laya imported'"
    probe += "; assert 'torch' not in sys.modules, 'torch imported'"
    finished = subprocess.run(
        [sys.executable, "-c", probe], cwd=BACKEND_ROOT, capture_output=True, text=True
    )
    assert finished.returncode == 0, finished.stderr


def test_fake_agent_satisfies_the_agent_protocol():
    assert isinstance(FakeAgent(), Agent)
    assert isinstance(LocalLayaAgent(model_dir="unused", device="cpu", load=False), Agent)


def test_fake_agent_returns_a_choice_that_was_offered_and_records_the_call():
    agent = FakeAgent(script={"piece_selection": "N"})
    options = {"P": "Pawn", "N": "Knight", "B": "Bishop"}

    result = agent.predict_choice("state text", "piece_selection", "Choose.", options)

    assert result.choice == "N"
    assert set(result.probabilities) == set(options)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert agent.question_ids == ["piece_selection"]
    assert agent.call_count == 1
    assert agent.calls == [("piece_selection", "state text", "Choose.", options)]


def test_fake_agent_consumes_a_scripted_list_one_entry_per_call():
    agent = FakeAgent(script={"destination_selection": ["e4", "d5"]})
    options = {"e4": "e4", "d5": "d5"}

    first = agent.predict_choice("s", "destination_selection", "c", options)
    second = agent.predict_choice("s", "destination_selection", "c", options)
    third = agent.predict_choice("s", "destination_selection", "c", options)

    # The third call has exhausted the script, so it takes the documented fallback:
    # the first offered option.
    assert [first.choice, second.choice, third.choice] == ["e4", "d5", "e4"]
    assert agent.call_count == 3


def test_fake_agent_falls_back_to_the_first_option_for_an_unscripted_question():
    """An absent, exhausted or unoffered script must still yield a legal answer."""
    agent = FakeAgent(script={"piece_selection": "Z"})

    result = agent.predict_choice("s", "piece_selection", "c", {"P": "Pawn", "N": "Knight"})

    assert result.choice == "P"
    assert agent.call_count == 1


def test_fake_agent_explicit_probabilities_put_all_mass_on_the_argmax():
    agent = FakeAgent(probabilities={"piece_selection": {"Q": 1.0}})

    result = agent.predict_choice("s", "piece_selection", "c", {"P": "Pawn", "Q": "Queen"})

    assert result.choice == "Q"
    assert result.probabilities == {"P": 0.0, "Q": 1.0}
