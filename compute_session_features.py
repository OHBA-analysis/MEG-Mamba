"""Precompute per-parcel conditioning features (token unigram + bigram).

For each session s and parcel c, from the whole session, compute:
  - unigram: normalised token-frequency histogram          (vocab,)
  - bigram : normalised token-transition matrix, flattened  (vocab*vocab,)

Each is ``sqrt``-transformed (variance-stabilising for the skewed compositional
counts), then a **frozen feature-wise standardisation** is applied — mean/std fit
on the TRAIN sessions only (the held-out/eval sessions do not contribute to the
stats, so there's no leakage), applied identically to everyone. The high-dim
bigram block is then PCA-reduced (PCA also fit on train) to keep it compact.

Output (``data/session_features.npz``):
  feat : (n_sessions, n_parcels, vocab + bigram_pca) float32   [unigram_std | bigram_pca]
  + the transform params (sqrt → standardise → PCA) so genuinely new sessions can
    be processed identically later.

Run in the mamba3 env (numpy/sklearn; no GPU needed):
    python compute_session_features.py --manifest data/manifest.pt
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def per_parcel_counts(tokens, vocab):
    """tokens (T, C) int -> unigram (C, vocab), bigram (C, vocab*vocab) frequencies."""
    C = tokens.shape[1]
    uni = np.zeros((C, vocab), np.float64)
    bi = np.zeros((C, vocab * vocab), np.float64)
    for c in range(C):
        col = tokens[:, c]
        uni[c] = np.bincount(col, minlength=vocab)[:vocab]
        pair = col[:-1].astype(np.int64) * vocab + col[1:].astype(np.int64)
        bi[c] = np.bincount(pair, minlength=vocab * vocab)[:vocab * vocab]
    uni /= uni.sum(1, keepdims=True) + 1e-9
    bi /= bi.sum(1, keepdims=True) + 1e-9
    return uni, bi


def featurize_session(tokens, transform):
    """Apply the frozen sqrt→standardize→PCA transform to one session's tokens.

    For zero-shot use of the released model on a new recording: turns a session's
    tokens into the per-parcel feature vectors the session-embedding MLP expects,
    using the released ``session_feature_transform.npz`` params.

    tokens:    (T, n_parcels) int.
    transform: the loaded ``session_feature_transform.npz`` (or any mapping with
               ``vocab, u_mu, u_sd, b_mu, b_sd, pca_components, pca_mean``).
    Returns (n_parcels, vocab + bigram_pca) float32 — wrap in ``[None]`` and pass to
    ``model.set_session_features`` to condition generation on this single session.
    """
    uni, bi = per_parcel_counts(np.asarray(tokens, np.int64), int(transform["vocab"]))
    u = (np.sqrt(uni) - transform["u_mu"]) / transform["u_sd"]            # (C, vocab)
    b = (np.sqrt(bi) - transform["b_mu"]) / transform["b_sd"]             # (C, vocab^2)
    b_pca = (b - transform["pca_mean"]) @ transform["pca_components"].T   # (C, bigram_pca)
    return np.concatenate([u, b_pca], axis=-1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.pt")
    p.add_argument("--output", default="data/session_features.npz")
    p.add_argument("--vocab", type=int, default=None, help="token vocab size (default: manifest vocab_size).")
    p.add_argument("--bigram-pca", type=int, default=200)
    args = p.parse_args()

    import torch
    manifest = torch.load(args.manifest, weights_only=False)
    entries = manifest["entries"]
    n_parcels = manifest["n_parcels"]
    vocab = args.vocab or int(manifest.get("vocab_size", 128))
    idx_of = lambda e: int(e["index"])  # noqa: E731
    n_sessions = max(idx_of(e) for e in entries) + 1

    U = np.zeros((n_sessions, n_parcels, vocab), np.float32)
    B = np.zeros((n_sessions, n_parcels, vocab * vocab), np.float32)
    present = []
    for i, e in enumerate(entries):
        s = idx_of(e)
        with h5py.File(e["path"], "r") as h:
            seg = np.asarray(h["tokens"][:], dtype=np.int64)    # whole session
        uni, bi = per_parcel_counts(seg, vocab)
        U[s], B[s] = uni, bi
        present.append(s)
        if (i + 1) % 50 == 0 or i + 1 == len(entries):
            print(f"  features {i+1}/{len(entries)}", flush=True)

    # sqrt transform (variance-stabilising for the compositional frequencies)
    U = np.sqrt(U)
    B = np.sqrt(B)

    # Frozen transforms fit on train sessions only, over (s,c) samples.
    holdout = set(json.loads(Path("data/holdout_sessions.json").read_text())["sessions"])
    train_s = np.array([s for s in present if s not in holdout])
    print(f"Fitting transforms on {len(train_s)} train sessions "
          f"({len(holdout & set(present))} held out of stats).")

    # unigram: per-bin standardise
    u_mu = U[train_s].reshape(-1, vocab).mean(0)
    u_sd = U[train_s].reshape(-1, vocab).std(0) + 1e-6
    U_std = (U - u_mu) / u_sd

    # bigram: per-bin standardise, then PCA (both fit on train)
    Btr = B[train_s].reshape(-1, vocab * vocab)
    b_mu = Btr.mean(0)
    b_sd = Btr.std(0) + 1e-6
    Btr_std = (Btr - b_mu) / b_sd
    from sklearn.decomposition import PCA
    k = min(args.bigram_pca, Btr_std.shape[1], Btr_std.shape[0])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Btr_std)
    del Btr, Btr_std
    B_pca = pca.transform((B.reshape(-1, vocab * vocab) - b_mu) / b_sd)
    B_pca = B_pca.reshape(n_sessions, n_parcels, k).astype(np.float32)
    print(f"bigram PCA: {k} comps explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% of (standardised) variance")

    feat = np.concatenate([U_std.astype(np.float32), B_pca], axis=-1)  # (S, P, vocab+k)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        feat=feat, vocab=vocab, n_parcels=n_parcels, bigram_pca=k,
        u_mu=u_mu, u_sd=u_sd, b_mu=b_mu, b_sd=b_sd,
        pca_components=pca.components_, pca_mean=pca.mean_,
        train_sessions=train_s, present=np.array(present),
    )
    print(f"feat {feat.shape} (unigram {vocab} + bigram_pca {k}) -> {args.output}")

    # Save the transform-only file
    transform_path = str(Path(args.output).with_name("session_feature_transform.npz"))
    np.savez(
        transform_path,
        vocab=vocab, n_parcels=n_parcels, bigram_pca=k,
        u_mu=u_mu, u_sd=u_sd, b_mu=b_mu, b_sd=b_sd,
        pca_components=pca.components_, pca_mean=pca.mean_,
    )
    print(f"transform params -> {transform_path}")


if __name__ == "__main__":
    main()
