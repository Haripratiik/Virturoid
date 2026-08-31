from virturoid.schemas.training_tip import ConfigScale, TipTrigger, TrainingTip, tip_from_budget_change


def test_tip_is_a_scoped_scaling_law_not_an_absolute_magic_value():
    tip = tip_from_budget_change(
        verdict="barely_moves",
        before={"generations": 30, "pop": 24, "steps": 320},
        after={"generations": 54, "pop": 36, "steps": 320},
        n_tokens=12,
        source_gene="quad-7",
    )
    assert tip.validate().ok
    assert tip.applies(stage="locomotion_search", verdict="barely_moves", n_tokens=14)
    assert not tip.applies(stage="locomotion_search", verdict="fell_over", n_tokens=14)
    assert not tip.applies(stage="locomotion_search", verdict="barely_moves", n_tokens=40)
    assert tip.apply({"generations": 10, "pop": 10, "steps": 100}) == {
        "generations": 18, "pop": 15, "steps": 100,
    }
    assert TrainingTip.from_dict(tip.to_dict()).to_dict() == tip.to_dict()


def test_tip_rejects_open_ended_or_absolute_mutations():
    tip = TrainingTip(
        id="unsafe",
        source_gene="g",
        trigger=TipTrigger(stage="locomotion_search", verdict="fell_over"),
        deltas=[ConfigScale(field="learning_rate", operation="set", factor=8.0)],
    )
    assert not tip.validate().ok
