"""Train MEG-Mamba.

Single-GPU training loop with bf16 AMP, linear-warmup → constant LR, best/latest
checkpointing, periodic snapshots, and resume.

Example:

    python train.py \
        --manifest data/manifest.pt \
        --d-model 256 --n-layers 4 --mimo-rank 4 \
        --seq-len 1000 --loss-start 50 \
        --batch-size 512 --lr 1e-4 --warmup-epochs 1 --epochs 15 \
        --snapshot-every 1 \
        --output-dir model
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

from dataset import create_dataloaders
from model import MEGMamba


def get_lr(step, warmup_steps, max_lr):
    """Linear warmup over ``warmup_steps`` then constant ``max_lr``."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    return max_lr


def save_checkpoint(path, model, optimizer, epoch, global_step, best_val):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state() if torch.cuda.is_available() else None
            ),
        },
        path,
    )


@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype):
    model.eval()
    # Weight each batch's mean by its row count so the final partial batch
    # (val_loader has no drop_last) doesn't get over-weighted. Every sequence is
    # the same length, so row-weighting recovers the exact token-level mean.
    tot_ce, tot_top1, tot_ent, tot_conf, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in val_loader:
        x = batch["input_seq"].to(device, non_blocking=True)
        y = batch["target_seq"].to(device, non_blocking=True)
        cid = batch["channel_id"].to(device, non_blocking=True)
        sid = batch["session_id"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x, cid, session_ids=sid)
            _, m = model.loss_from_output(logits, y)
        nb = x.shape[0]
        tot_ce += m["ce"] * nb
        tot_top1 += m["top1_acc"] * nb
        tot_ent += m["entropy"] * nb
        tot_conf += m["top1_prob"] * nb
        n += nb
    model.train()
    return {
        "val_ce": tot_ce / max(1, n),
        "val_top1": tot_top1 / max(1, n),
        "val_entropy": tot_ent / max(1, n),
        "val_top1_prob": tot_conf / max(1, n),
    }


def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--manifest", type=str, default="data/manifest.pt")
    p.add_argument("--n-parcels", type=int, default=100)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=1000)  # 4 s @ 250 Hz

    # Precomputed session feature conditioning and held-out session set
    p.add_argument("--session-features", default="data/session_features.npz",
                   help="per-(session,parcel) feature table (token unigram+bigram) "
                        "built by compute_session_features.py — the input to "
                        "the embedding MLP.")
    p.add_argument("--holdout-sessions", default="test",
                   help="'test' (data/holdout_sessions.json) or a comma-separated "
                        "list of session indices. Fully held out of training; they "
                        "ARE the validation set (scored on their whole recording "
                        "each epoch).")

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--mimo-rank", type=int, default=4)  # kernel requires >=4
    p.add_argument("--loss-start", type=int, default=50)
    p.add_argument("--sess-emb-hidden-dim", type=int, default=128, help="hidden width of the session-embedding MLP (features -> session emb).")

    # Optimizer
    p.add_argument("--batch-size", type=int, default=512, help="channel-sequences per step (= per-step kernel batch rows)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Override warmup length in steps (else "
                        "warmup_epochs * steps_per_epoch). Useful for short "
                        "chained jobs so the first one reaches peak LR sooner.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast (train in fp32).")

    # IO
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=500,
                   help="Steps between mid-epoch rolling saves to latest.pt "
                        "(crash safety; 0 disables). Overwrites one file, so it "
                        "costs no extra disk. On --resume the in-progress epoch "
                        "re-runs from its start with the saved (warmed) weights.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (Python/NumPy/torch+CUDA) for reproducible data "
                        "order + weight init. NB: the Mamba-3 GPU kernels are not "
                        "bitwise-deterministic, so runs are closely but not "
                        "bit-identically reproducible.")
    args = p.parse_args()

    # Reproducibility: seed Python / NumPy / torch (+ CUDA) before any RNG use
    # (weight init, DataLoader shuffle). Bit-identical reproduction isn't possible
    # — the Mamba-3 TileLang kernels are nondeterministic on GPU — but this fixes
    # the data order and weight init so runs are closely reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if args.no_amp else torch.bfloat16
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Mamba-3 MIMO kernel requires seq_len divisible by chunk_size (8).
    if args.seq_len % 8 != 0:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) must be a multiple of 8 for the "
            f"Mamba-3 MIMO kernel (chunk_size=8)."
        )

    # Data: train on the non-held-out sessions (full time); the held-out
    # sessions are the validation set (zero-shot, via feature conditioning).
    manifest = torch.load(args.manifest, weights_only=False)

    # take data dims from the manifest so the model always matches the tokens
    args.n_parcels = manifest["n_parcels"]
    args.vocab_size = int(manifest.get("vocab_size", args.vocab_size))

    # Held-out session set: 'test' → data/holdout_sessions.json, else a list.
    if args.holdout_sessions == "test":
        holdout_sessions = json.loads(Path("data/holdout_sessions.json").read_text())["sessions"]
    else:
        holdout_sessions = [int(x) for x in args.holdout_sessions.split(",")]

    # Embedding-MLP feature table (per session, parcel) — zero-shot to held-out sessions.
    cf = np.load(args.session_features)
    cond_table = torch.tensor(cf["feat"], dtype=torch.float32)   # (n_sess, n_parcels, F)
    sess_emb_feat_dim = int(cond_table.shape[-1])
    print(f"Features: {args.session_features} -> table {tuple(cond_table.shape)}")

    train_loader, val_loader = create_dataloaders(
        manifest, holdout_sessions,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    # Model
    model_kwargs = dict(
        n_parcels=args.n_parcels,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        mimo_rank=args.mimo_rank,
        is_outproj_norm=True,
        loss_start=args.loss_start,
        sess_emb_feat_dim=sess_emb_feat_dim,
        sess_emb_hidden_dim=args.sess_emb_hidden_dim,
    )
    model = MEGMamba(**model_kwargs).to(device)
    model.set_session_features(cond_table)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MEGMamba: {n_params/1e6:.2f}M params, {len(holdout_sessions)} sessions "
          f"held out (validation), sess_emb_feat_dim={sess_emb_feat_dim}")

    model_config = dict(model_kwargs)
    (out / "model_config.json").write_text(json.dumps(model_config, indent=2))
    (out / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else args.warmup_epochs * steps_per_epoch

    # Resume
    start_epoch, global_step, best_val = 0, 0, float("inf")
    metrics_log = []
    latest_path = out / "latest.pt"
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]      # epochs completed so far (1-indexed)
        global_step = ckpt["global_step"]
        best_val = ckpt["best_val"]
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu())
        if (out / "metrics.json").exists():
            metrics_log = json.loads((out / "metrics.json").read_text())
            # Keep only completed-epoch rows. If a crash landed between the
            # metrics.json write and the end-of-epoch checkpoint, metrics_log can
            # hold a row for the epoch we're about to re-run; drop it so that
            # epoch isn't logged twice.
            metrics_log = metrics_log[:start_epoch]
        print(f"Resuming at epoch {start_epoch + 1} (global_step {global_step})")

    print(f"{steps_per_epoch} steps/epoch, {warmup_steps} warmup steps, LR {args.lr} (warmup → constant)")

    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):   # epochs are 1-indexed (1..N)
        t0 = time.time()
        run_ce, run_top1, gnorm_sum, gnorm_max = 0.0, 0.0, 0.0, 0.0
        lr = args.lr
        for it, batch in enumerate(train_loader):
            lr = get_lr(global_step, warmup_steps, args.lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            x = batch["input_seq"].to(device, non_blocking=True)
            y = batch["target_seq"].to(device, non_blocking=True)
            cid = batch["channel_id"].to(device, non_blocking=True)
            sid = batch["session_id"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x, cid, session_ids=sid)
                loss, m = model.loss_from_output(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            run_ce += m["ce"]
            run_top1 += m["top1_acc"]
            g = float(gnorm)
            gnorm_sum += g
            gnorm_max = max(gnorm_max, g)

            # Mid-epoch crash-safety: overwrite latest.pt every N steps. Store
            # `epoch - 1` as epochs-completed so --resume re-runs this still
            # in-progress epoch from its start with the warmed weights (the
            # DataLoader can't resume mid-epoch); global_step is preserved so
            # the LR-warmup schedule continues seamlessly.
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(latest_path, model, optimizer, epoch - 1, global_step, best_val)

            if it % args.log_every == 0:
                print(f"e{epoch} it{it}/{steps_per_epoch} "
                      f"ce {m['ce']:.4f} top1 {m['top1_acc']:.3f} "
                      f"lr {lr:.2e} gnorm {float(gnorm):.2f}")

        # Validation = the held-out SESSIONS (unseen session × unseen time).
        val = evaluate(model, val_loader, device, amp_dtype)
        train_ce = run_ce / max(1, steps_per_epoch)
        train_top1 = run_top1 / max(1, steps_per_epoch)
        gnorm_mean = gnorm_sum / max(1, steps_per_epoch)
        dt = time.time() - t0
        print(f"== epoch {epoch}: train_ce {train_ce:.4f} train_top1 {train_top1:.3f} "
              f"heldout_ce {val['val_ce']:.4f} heldout_top1 {val['val_top1']:.3f} "
              f"heldout_ent {val['val_entropy']:.3f} gnorm~{gnorm_mean:.2f} "
              f"({dt:.0f}s) ==")
        metrics_log.append({
            "epoch": epoch,
            # train (mean over the epoch)
            "train_ce": train_ce,
            "train_top1": train_top1,
            "train_ppl": float(np.exp(train_ce)),
            "grad_norm_mean": gnorm_mean,
            "grad_norm_max": gnorm_max,
            "lr": lr,
            # held-out sessions (zero-shot validation)
            "heldout_ce": val["val_ce"],
            "heldout_top1": val["val_top1"],
            "heldout_ppl": float(np.exp(val["val_ce"])),
            "heldout_entropy": val["val_entropy"],
            "heldout_top1_prob": val["val_top1_prob"],
            "time_s": dt,
        })
        (out / "metrics.json").write_text(json.dumps(metrics_log, indent=2))

        save_checkpoint(latest_path, model, optimizer, epoch, global_step, best_val)

        if val["val_ce"] < best_val:
            best_val = val["val_ce"]
            save_checkpoint(out / "best_model.pt", model, optimizer, epoch, global_step, best_val)
            print(f"  new best val_ce {best_val:.4f}")

        if args.snapshot_every and epoch % args.snapshot_every == 0:
            save_checkpoint(out / f"snapshot_e{epoch}.pt", model, optimizer, epoch, global_step, best_val)

    print(f"Done. best val_ce {best_val:.4f}")


if __name__ == "__main__":
    main()
