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
                      topo_bias: bool = False, percept=None):
    """(NT, F) obs -> (action_means (NT,), pooled_embed (H,)). ``att`` holds We/be/Wq/Wk/Wv/Wo/Wh/bh
    (+ optional Wfilm, Wtopo, Wrange/brange). ``xp`` is numpy or jax.numpy. ``hop`` is an (NT, NT)
    hop-distance matrix (only read when ``topo_bias``).

    ``percept`` (an already-packed length-``PERCEPT_DIM`` vector: rangefinder distances + [vx, wz] command,
    or a phase clock) appends ONE global perception token — encoded straight into HIDDEN space via Wrange —
    that the joints attend to, so the body can act on what it SENSES / on a gait clock (Siekmann-style). This
    mirrors ``MorphPolicy.act``'s perception path EXACTLY so a GPU-trained policy deploys byte-identically on
    the CPU. ``percept=None`` is byte-identical to the proprioception-only forward (perception-blind), so every
    existing policy behaves exactly as before. The percept token carries no action and does not enter the
    pooled critic embedding (both slice to ``[:nj]``)."""
    e = xp.tanh(obs @ att["We"] + att["be"])
    if film:                                              # FiLM: (1+gamma)*e + beta from the obs datasheet
        gb = obs @ att["Wfilm"]
        e = (1.0 + gb[:, :hidden]) * e + gb[:, hidden:]
    nj = obs.shape[0]                                      # number of JOINT tokens (excludes the percept token)
    if percept is not None:                               # global perception/clock token (see MorphPolicy.act)
        pt = xp.tanh(xp.asarray(percept) @ att["Wrange"] + att["brange"])   # (H,) in hidden space
        e = xp.concatenate([e, pt[None, :]], axis=0)                        # (nj+1, H): joints + perception
    q, k, v = e @ att["Wq"], e @ att["Wk"], e @ att["Wv"]
    s = (q @ k.T) / xp.sqrt(hidden)
    if topo_bias and hop is not None:                     # structure-aware bias theta_{d(i,j)} on hop distance
        tb = att["Wtopo"][xp.clip(hop, 0, att["Wtopo"].shape[0] - 1)]       # (nj, nj) joint-only bias
        if e.shape[0] != nj:                              # pad zeros for the percept row/col (it gets no bias)
            tb = xp.pad(tb, ((0, e.shape[0] - nj), (0, e.shape[0] - nj)))
        s = s + tb
    s = s - s.max(axis=1, keepdims=True)
    w = xp.exp(s)
    w = w / w.sum(axis=1, keepdims=True)
    u = xp.tanh(e + (w @ v) @ att["Wo"])
    out = xp.tanh(u @ att["Wh"] + att["bh"])[:, 0]
    return out[:nj], u[:nj].mean(0)                        # drop the percept token's action + pooled slot
