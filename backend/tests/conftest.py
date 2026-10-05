"""Shared pytest fixtures.

Nothing here loads the real checkpoint. Tests install a `FakeAgent` in `app.state`
before the app starts, so `app.main.load_model` finds a model already in place and returns
without reading the 842 MB directory. Only `get_db` is overridden: `get_agent` is left
alone on purpose, so the real `app.deps.get_agent` is what decides between serving a move
and answering `503`.
"""

from __future__ import annotations

from collections.abc import Iterator

import chess
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app import state
from app.db import Base, get_db, make_engine
from app.main import create_app
from app.models import Game
from tests.support import FakeAgent


@pytest.fixture
def start_board() -> chess.Board:
    """Return a fresh board at the standard starting position.

    Returns:
        A `chess.Board` with no moves played.
    """
    return chess.Board()


@pytest.fixture
def agent() -> Iterator[FakeAgent]:
    """Install a `FakeAgent` as the process-wide agent for the duration of a test.

    Set up before `client` starts the app, which is the whole reason the fake lives in
    `app.state` rather than behind a dependency override: `load_model` checks the state
    before it reaches for the checkpoint. Yielding it separately is what lets a test
    script a specific line, such as the two black moves that make fool's mate possible.

    Yields:
        The `FakeAgent` every request will be answered by.
    """
    fake = FakeAgent()
    state.set_agent(fake)
    yield fake
    state.clear_agent()


@pytest.fixture
def client(tmp_path, agent) -> Iterator[TestClient]:
    """Yield a `TestClient` wired to a throwaway database and the `FakeAgent`.

    Overriding `get_db` keeps tests off the developer's real `chessbot.db`; overriding
    `get_agent` would be wrong here, because it would route around the `503` that
    `app.deps.get_agent` raises when no model is loaded -- and that branch is part of what
    this suite exists to pin. The dependency reads `app.state`, which already holds the
    fake, so leaving it alone costs nothing and proves more.

    The lifespan still runs, so `init_db` is exercised. `load_model` then finds an agent
    already installed and skips the checkpoint.

    Args:
        tmp_path: pytest's temporary directory, which holds the test database.
        agent: The `FakeAgent` fixture, installed before the app starts.

    Yields:
        A `TestClient` for the app.
    """
    engine = make_engine(f"sqlite:///{(tmp_path / 'games.db').as_posix()}")
    Base.metadata.create_all(bind=engine)
    testing_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override_get_db() -> Iterator[Session]:
        session = testing_session()
        try:
            yield session
        finally:
            session.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.fixture
def seed_game(tmp_path):
    """Return a callable that inserts a `Game` row straight into the test database.

    Some states are not reachable through the API. A position that is not the human's
    turn is one of them: the human's ply and the bot's reply always land in a single
    request, so no sequence of requests can leave a game on the bot's side. Writing the
    row is the honest way to reach the guard -- mocking the board would prove nothing
    about the ordering the guard exists to enforce.

    Args:
        tmp_path: pytest's temporary directory, shared with the `client` fixture, so the
            row lands in the database that client is talking to.

    Returns:
        A callable taking `Game` column values and returning the persisted row.
    """
    url = f"sqlite:///{(tmp_path / 'games.db').as_posix()}"

    def seed(**fields) -> Game:
        engine = make_engine(url)
        Base.metadata.create_all(bind=engine)
        try:
            with sessionmaker(bind=engine)() as session:
                game = Game(**fields)
                session.add(game)
                session.commit()
                session.refresh(game)
                session.expunge(game)
            return game
        finally:
            engine.dispose()

    return seed
