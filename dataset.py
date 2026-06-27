"""Dataset and Data Loaders.

Each example is **one channel's** token window: ``__getitem__`` returns
``input_seq`` / ``target_seq`` of shape ``(seq_len,)`` int64 (next-token targets
are the input shifted by one), the integer ``channel_id``, and the integer
``session_id`` (the session index, for the model's session embedding).
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenizedMEGDataset(Dataset):
    """Per-channel windowed dataset over tokenized ``.h5`` sessions.

    Example index ``i`` maps to ``(window, channel)`` via
    ``window, channel = divmod(i, n_parcels)``.
    """

    def __init__(self, entries, seq_len, *, token_store):
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")

        self.seq_len = seq_len
        self.n_parcels = token_store[0].shape[1]
        self.session_ids = [int(e["index"]) for e in entries]
        self.token_store = token_store

        # Window starts over the whole recording (non-overlapping; stride =
        # seq_len). A window needs seq_len + 1 tokens (the extra one is the final
        # next-token target).
        self.windows = []
        for entry_idx, e in enumerate(entries):
            max_start = e["n_samples"] - seq_len - 1
            for start in range(0, max_start + 1, seq_len):
                self.windows.append((entry_idx, start))

    def __len__(self):
        return len(self.windows) * self.n_parcels

    def __getitem__(self, idx):
        window_idx, channel = divmod(idx, self.n_parcels)
        entry_idx, start = self.windows[window_idx]
        stop = start + self.seq_len + 1
        seq = self.token_store[entry_idx][start:stop, channel].long()
        return {
            "input_seq": seq[:-1],
            "target_seq": seq[1:],
            "channel_id": channel,
            "session_id": self.session_ids[entry_idx],
        }


def _preload_token_store(entries):
    """Load every session's tokens fully into RAM as int16 (one h5 read per file).

    Returns a list indexed by entry_idx of (n_samples, n_parcels) int16 tensors.
    """
    store = []
    for e in entries:
        with h5py.File(e["path"], "r") as h:
            arr = h["tokens"][:]  # (T, n_parcels) uint8
        store.append(torch.from_numpy(arr.astype(np.int16)))
    n_gb = sum(t.numel() for t in store) * 2 / 1e9
    print(f"Preloaded {len(store)} sessions into RAM ({n_gb:.1f} GB int16)")
    return store


def create_dataloaders(
    manifest_path,
    holdout_sessions,
    batch_size=512,
    seq_len=1000,
):
    """Build train + held-out-session DataLoaders from a manifest.

    The split is by session. ``holdout_sessions`` are fully removed from
    training; both sets window their whole recordings. The validation set is
    the held-out sessions.

    ``batch_size`` is the number of channel-sequences per step (= per-step
    kernel batch rows).

    Returns ``(train_loader, val_loader)``.
    """
    manifest = (
        torch.load(manifest_path, weights_only=False)
        if isinstance(manifest_path, (str, Path))
        else manifest_path
    )
    entries = manifest["entries"]

    def _idx(e):
        return int(e["index"])

    hold = {int(s) for s in holdout_sessions}
    train_entries = [e for e in entries if _idx(e) not in hold]
    hold_entries = [e for e in entries if _idx(e) in hold]
    if not hold_entries:
        raise ValueError("holdout_sessions matched no sessions in the manifest.")

    train_store = _preload_token_store(train_entries)
    train_ds = TokenizedMEGDataset(train_entries, seq_len, token_store=train_store)
    val_store = _preload_token_store(hold_entries)
    val_ds = TokenizedMEGDataset(hold_entries, seq_len, token_store=val_store)

    print(
        f"Train: {len(train_entries)} sessions (full time) | "
        f"Val (held-out sessions): {len(hold_entries)} | "
        f"{len(train_ds)} train / {len(val_ds)} val channel-sequences "
        f"(seq_len={seq_len}, batch={batch_size} rows)"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
