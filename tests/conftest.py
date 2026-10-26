import os

import pytest

from devcopilot.config import get_settings
from devcopilot.world.seed import seed


@pytest.fixture(scope="session")
def seeded_workspace(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("workspace")
    hashes = seed(workspace)
    return workspace, hashes


def _use_workspace(monkeypatch, workspace):
    monkeypatch.setenv("DEVCOPILOT_WORKSPACE_DIR", str(workspace))
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def use_seeded_workspace(seeded_workspace, monkeypatch):
    """Point every test at a shared seeded workspace. Tests must not modify it.

    Also neutralize this machine's settings: no .env, no DEVCOPILOT_* or Jev key from
    the shell. Tests see the code's defaults plus the overrides below, so no test reaches
    a real Jev server or a model on another host, whatever the machine is configured for.
    """
    for name in [k for k in os.environ if k.startswith("DEVCOPILOT_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEVCOPILOT_DOTENV", "0")
    monkeypatch.setenv("DEVCOPILOT_JEV_FOR", "")
    # Unroutable: a test that forgets to fake the decision server fails fast, offline.
    monkeypatch.setenv("DEVCOPILOT_JEV_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("DEVCOPILOT_JEV_MODEL", "test-decider")
    monkeypatch.setenv("DEVCOPILOT_ROUTES", "")
    monkeypatch.setenv("DEVCOPILOT_MEMORY", "false")
    monkeypatch.setenv("DEVCOPILOT_VALIDATE_REPORTS", "false")
    _use_workspace(monkeypatch, seeded_workspace[0])
    yield
    get_settings.cache_clear()


@pytest.fixture
def fresh_workspace(tmp_path, monkeypatch):
    """A private workspace for tests that write (issues, branches)."""
    workspace = tmp_path / "workspace"
    seed(workspace)
    _use_workspace(monkeypatch, workspace)
    return workspace


@pytest.fixture
def commits(seeded_workspace):
    return seeded_workspace[1]
