import pytest


def test_feature_mismatch_is_reported_instead_of_silent(monkeypatch):
    from virturoid.services import morph_trainer
    from virturoid.services.morph_policy import MorphPolicy

    monkeypatch.setattr(morph_trainer, "forward_score", lambda gene, policy, **kwargs: 0.25)
    prior = MorphPolicy(7, seed=1)
    with pytest.warns(RuntimeWarning, match="warm-start rejected"):
        policy, history = morph_trainer.train_morph_es(
            [object()], feature_dim=9, generations=0, pop=2, init_policy=prior, eval_backend="serial"
        )
    assert policy.feature_dim == 9
    assert history == [0.25]
