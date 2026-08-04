import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "miner-utils" / "src"))

from stratum_client import StratumClient
from miner.miner_gpu import inject_pool_difficulty_into_password


def test_inject_pool_difficulty_into_password_replaces_existing_directive():
    assert inject_pool_difficulty_into_password("x,d=1000", 500) == "x,d=500"
    assert inject_pool_difficulty_into_password("d=1000", 750) == "d=750"


def test_inject_pool_difficulty_into_password_appends_when_no_directive():
    assert inject_pool_difficulty_into_password("x", 600) == "x,d=600"
    assert inject_pool_difficulty_into_password("password", 1200) == "password,d=1200"
    assert inject_pool_difficulty_into_password("", 250) == "d=250"


def test_subscribe_timeout_is_tolerated_for_pearl(monkeypatch):
    client = StratumClient(
        pool_url="stratum+tcp://example:3333",
        username="wallet.worker",
        password="x",
        algorithm="pearl",
    )

    monkeypatch.setattr(client, "_send_fire_and_forget", lambda *args, **kwargs: True)

    def fake_send_request(method, params, timeout=10.0):
        if method == "mining.subscribe":
            return None
        if method == "mining.authorize":
            return {"result": True}
        return {"result": True}

    monkeypatch.setattr(client, "_send_request", fake_send_request)

    assert client._subscribe_and_authorize() is True
    assert client._subscribed is True
    assert client._authorized is True
