from belladonna import hello


def test_hello() -> None:
    assert hello() == "Belladonna is ready."
