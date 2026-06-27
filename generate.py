"""Generate token rollouts for the held-out eval sessions.

The held-out sessions (``data/holdout_sessions.json``) never appear in training.
For each we take a random ``n_context + n_generate`` window of its recording,
condition on the first ``n_context`` real tokens of that window plus the session's
channel + session ids (which index the feature table), then autoregressively generate
``n_generate`` more tokens per channel. Saves one ``*_gen.npz`` per session.

Run in the `mamba3` env:

    python generate.py --checkpoint model/weights.pt \
        --session-features data/session_features.npz \
        --output-dir ~/projects/meg-mamba/analysis/generated \
        --n-context 1000 --n-generate 7500 --session-batch 4
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from model import MEGMamba


def _safe_name(sid):
    return sid.replace("/", "_").replace("\\", "_")


def _filter_logits(logits, top_k=0, top_p=1.0):
    """Top-k / nucleus (top-p) filtering along the last (vocab) axis.

    logits: (..., V). Returns logits with filtered entries set to -inf.
    """
    if top_k and top_k < logits.shape[-1]:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    return logits


def _sample(logits, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Sample one token per row. logits (M, V) → (M,)."""
    logits = logits / max(temperature, 1e-6)
    logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    context,
    channel_ids,
    n_generate,
    session_ids,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    seed=0,
):
    """Per-channel autoregressive rollout with O(1)-per-step state caching.

    Args:
        context:     (M, n_ctx) int tokens — M independent channel-sequences.
        channel_ids: (M,) int channel index for each row.
        n_generate:  number of new tokens to generate per row.
        session_ids: (M,) int session index (indexes the feature table).
        temperature / top_k / top_p: sampling controls (defaults = plain
            multinomial at temperature 1.0).

    Returns:
        (M, n_ctx + n_generate) int tokens (context followed by generations).
    """
    from mamba_ssm.utils.generation import InferenceParams

    model.eval()
    device = next(model.parameters()).device
    context = context.to(device).long()  # (M, n_ctx)
    channel_ids = channel_ids.to(device).long()  # (M,)
    M, n_ctx = context.shape
    session_ids = session_ids.to(device).long()
    gen = torch.Generator(device=device).manual_seed(seed)

    total_len = n_ctx + n_generate
    inference_params = InferenceParams(max_seqlen=total_len, max_batch_size=M)

    # Prefill: process the whole context (seqlen_offset stays 0 → full forward
    # path populates each row's conv/SSM cache with the end-of-context state).
    logits = model(
        context,
        channel_ids,
        inference_params=inference_params,
        session_ids=session_ids,
    )  # (M, n_ctx, V)
    inference_params.seqlen_offset = n_ctx  # → step mode hereafter

    next_tok = _sample(logits[:, -1], temperature, top_k, top_p, gen)  # (M,)
    out = [next_tok]
    for _ in range(n_generate - 1):
        logits = model(
            next_tok.unsqueeze(1),
            channel_ids,
            inference_params=inference_params,
            session_ids=session_ids
        )  # (M, 1, V)
        inference_params.seqlen_offset += 1
        next_tok = _sample(logits[:, 0], temperature, top_k, top_p, gen)
        out.append(next_tok)

    generated = torch.stack(out, dim=1)  # (M, n_generate)

    return torch.cat([context, generated], dim=1)  # (M, total_len)


def load_model(checkpoint, device, session_features="data/session_features.npz"):
    config = json.loads((Path(checkpoint).parent / "model_config.json").read_text())
    model = MEGMamba(**config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model = model.to(device).eval()
    table = torch.tensor(np.load(session_features)["feat"], dtype=torch.float32)
    model.set_session_features(table)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="data/manifest.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-context", type=int, default=1000)   # 4 s @ 250 Hz
    p.add_argument("--n-generate", type=int, default=7500)  # ~30 s self-conditioned
    p.add_argument("--session-batch", type=int, default=4,
                   help="held-out sessions per chunk; kernel batch = session_batch x n_parcels.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--session-features", default="data/session_features.npz",
                   help="embedding-MLP feature table (built by compute_session_features.py).")
    args = p.parse_args()

    # The MIMO kernel prefills the whole context in one chunked pass, so the
    # context length must be a multiple of chunk_size (8) — same constraint
    # train.py enforces on --seq-len. (n_generate is unconstrained: it is rolled
    # out one token at a time through the step kernel.)
    if args.n_context % 8 != 0:
        raise SystemExit(
            f"--n-context ({args.n_context}) must be a multiple of 8 for the "
            f"Mamba-3 MIMO kernel (chunk_size=8)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, session_features=args.session_features)
    C = model.n_parcels

    manifest = torch.load(args.manifest, weights_only=False)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    holdout_sessions = json.loads(Path("data/holdout_sessions.json").read_text())["sessions"]
    total_len = args.n_context + args.n_generate
    eval_set = set(holdout_sessions)
    entries = [(int(e["index"]), e) for e in manifest["entries"] if int(e["index"]) in eval_set]
    print(f"Generating for {len(entries)} eval sessions "
          f"(n_context={args.n_context}, n_generate={args.n_generate}, "
          f"session_batch={args.session_batch})")

    n_done = 0
    for chunk_idx, i in enumerate(range(0, len(entries), args.session_batch)):
        chunk = entries[i:i + args.session_batch]
        contexts, reals, sess, sids = [], [], [], []
        for idx, e in chunk:
            with h5py.File(e["path"], "r") as h:
                T = h["tokens"].shape[0]
                if T < total_len:
                    print(f"  skip {e['session']} ({T} tokens < {total_len})")
                    continue
                start = int(np.random.RandomState(args.seed + idx).randint(0, T - total_len + 1))
                seg = np.asarray(h["tokens"][start:start + total_len], dtype=np.int64)  # (total_len, C)
            contexts.append(seg[:args.n_context].T)  # (C, n_ctx)
            reals.append(seg)  # (total_len, C)
            sess.append(idx)
            sids.append(e["session"])
        if not contexts:
            continue

        B = len(contexts)
        context = torch.from_numpy(np.stack(contexts)).reshape(B * C, args.n_context)  # (B*C, n_ctx)
        channel_ids = torch.arange(C).repeat(B)  # (B*C,)
        session_ids = torch.tensor(sess).repeat_interleave(C)  # (B*C,) session idx per row

        gen = generate(
            model,
            context,
            channel_ids,
            args.n_generate,
            session_ids=session_ids,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed + chunk_idx,  # independent sampling stream per chunk
        )
        gen = gen.cpu().numpy().reshape(B, C, total_len)  # (B, C, total_len)

        for j, sid in enumerate(sids):
            # _safe_name flattens any dataset-prefix slash for the on-disk filename
            np.savez(
                out / f"{_safe_name(sid)}_gen.npz",
                generated=gen[j].T.astype(np.int32),  # (total_len, C)
                real=reals[j].astype(np.int32),  # (total_len, C)
                session_id=sid, n_context=args.n_context,
                sfreq=manifest.get("sfreq", 250),
            )
            n_done += 1
        print(f"  [{n_done}/{len(entries)}] {sids[0]}…{sids[-1]}")

    print(f"Done — {n_done} eval sessions → {out}")


if __name__ == "__main__":
    main()
