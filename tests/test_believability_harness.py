import json


def _fixture(root, payload):
    from virturoid.services.believability_harness import ROBOTS, VIEWS
    root.mkdir()
    for robot in ROBOTS:
        for view in VIEWS:
            (root / f"{robot}_{view}.png").write_bytes(payload + f"{robot}:{view}".encode())


def test_protocol_is_blind_fixed_and_repeatable(tmp_path):
    from virturoid.services.believability_harness import prepare_protocol
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    _fixture(baseline, b"old")
    _fixture(candidate, b"new")
    first = prepare_protocol(baseline, candidate, tmp_path / "one", seed=17)
    second = prepare_protocol(baseline, candidate, tmp_path / "two", seed=17)
    ballot = json.loads((tmp_path / "one" / "ballot.json").read_text())
    assert first["protocol_id"] == second["protocol_id"]
    assert first["n_pairs"] == 18
    assert all("candidate_side" not in pair for pair in ballot["pairs"])


def test_scoring_reports_preference_and_wilson_ci():
    from virturoid.services.believability_harness import PROTOCOL_VERSION, score_votes
    key = {"protocol": PROTOCOL_VERSION, "protocol_id": "p", "candidate_side_by_pair": {"p1": "A", "p2": "B"}}
    votes = ([{"pair_id": "p1", "choice": "A"}] * 8 + [{"pair_id": "p1", "choice": "B"}] * 2
             + [{"pair_id": "p2", "choice": "tie"}, {"pair_id": "bad", "choice": "A"}])
    result = score_votes(key, votes)
    assert result["candidate_preference"] == 0.8
    assert result["candidate_wins"] == 8 and result["baseline_wins"] == 2
    assert result["ties"] == 1 and result["invalid_votes"] == 1
    assert result["ci95"][0] < 0.8 < result["ci95"][1]
