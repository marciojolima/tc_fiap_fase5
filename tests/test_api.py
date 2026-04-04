from serving.app import healthcheck


def test_healthcheck_returns_ok() -> None:
    assert healthcheck() == {"status": "ok"}
