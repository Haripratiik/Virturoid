"""The shared morph-attention forward — ONE implementation the CPU policy and the GPU (Flax) trainer both run.

The MorphPolicy contract is "same forward on CPU (NumPy) and GPU (jax), so trained weights transfer to any
body." That only holds if the two forwards are byte-identical math; keeping two hand-written copies invites
drift (a transcription bug = a policy that trains on GPU but mis-controls on CPU). This function is that single
source of truth, written array-library-agnostic: pass ``xp=numpy`` (CPU) or ``xp=jax.numpy`` (GPU/Flax).

It covers the proprioception-only token forward + the Phase-5 tokenizer upgrades (FiLM joint-attribute
conditioning, topology-aware attention bias), all identity when their weights are zero. The perception token
(a global range/command token) stays in ``MorphPolicy.act`` — training is proprioception-only — so this is the
exact common core the trainer's ``policy_mean_pool`` and the CPU ``act`` share.
"""

from __future__ import annotations


def attention_forward(att: dict, obs, hidden: int, *, xp, film: bool = False, hop=None,
                      topo_bias: bool = False):
    """(NT, F) obs -> (action_means (NT,), pooled_embed (H,)). ``att`` holds We/be/Wq/Wk/Wv/Wo/Wh/bh
    (+ optional Wfilm, Wtopo). ``xp`` is numpy or jax.numpy. ``hop`` is an (NT, NT) hop-distance matrix
    (only read when ``topo_bias``). Mirrors MorphPolicy.act's proprioception-only path exactly."""
    e = xp.tanh(obs @ att["We"] + att["be"])
    if film:                                              # FiLM: (1+gamma)*e + beta from the obs datasheet
        gb = obs @ att["Wfilm"]
        e = (1.0 + gb[:, :hidden]) * e + gb[:, hidden:]
    q, k, v = e @ att["Wq"], e @ att["Wk"], e @ att["Wv"]
    s = (q @ k.T) / xp.sqrt(hidden)
    if topo_bias and hop is not None:                     # structure-aware bias theta_{d(i,j)} on hop distance
        s = s + att["Wtopo"][xp.clip(hop, 0, att["Wtopo"].shape[0] - 1)]
    s = s - s.max(axis=1, keepdims=True)
    w = xp.exp(s)
    w = w / w.sum(axis=1, keepdims=True)
    u = xp.tanh(e + (w @ v) @ att["Wo"])
    out = xp.tanh(u @ att["Wh"] + att["bh"])[:, 0]
    return out, u.mean(0)
