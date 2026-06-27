"""MEG-Mamba: an autoregressive model for tokenized MEG.

Each channel is modelled independently by a shared Mamba-3 backbone.

Two terms are added to the token embedding at the input:
- a learned per-parcel **channel embedding**.
- a **feature-derived session embedding** produced by a small MLP
  from precomputed per-(session,parcel) token features (unigram + bigram;
  see compute_session_features.py).

Pipeline:

    tokens (N, L) int  +  channel_ids (N,)  +  session_ids (N,)
      → token_emb(x) + channel_emb[cid] + session_MLP(session_features[sid, cid])  (N, L, d_model)
      → [ Mamba-3 (MIMO) + SwiGLU FFN, RMSNorm, residual ] * N_layers  (N, L, d_model)
      → final RMSNorm → shared Linear head  (N, L, V)

Shapes:
    input  tokens : (N, L) int64,   channel_ids/session_ids : (N,) int64
    output logits : (N, L, V) float
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba3
from mamba_ssm.ops.triton.layer_norm import RMSNorm

# Upstream Mamba3.step() expects 2D (B, D) input but forward() passes 3D
# (B, 1, D); patch it to squeeze/unsqueeze so single-step generation works.
_original_mamba3_step = Mamba3.step


def _patched_mamba3_step(self, u, *args, **kwargs):
    squeeze = u.dim() == 3
    if squeeze:
        u = u.squeeze(1)
    out = _original_mamba3_step(self, u, *args, **kwargs)
    if squeeze:
        out = (
            (out[0].unsqueeze(1),) + out[1:]
            if isinstance(out, tuple)
            else out.unsqueeze(1)
        )
    return out


Mamba3.step = _patched_mamba3_step


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block (Llama/PaLM-style gated FFN).

    out = W_out( SiLU(a) * b ),  where (a, b) = W_gate(x).chunk(2).
    With d_ff ≈ 4·d_model/3, params match a standard 2·d_model GELU FFN.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, 2 * d_ff)
        self.w_out = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.w_gate(x).chunk(2, dim=-1)
        return self.w_out(F.silu(a) * b)


class MEGMamba(nn.Module):
    """MEG-Mamba."""

    def __init__(
        self,
        n_parcels=100,
        vocab_size=128,
        d_model=256,
        n_layers=4,
        d_state=64,
        expand=2,
        headdim=64,
        mimo_rank=4,
        is_outproj_norm=True,
        loss_start=50,
        sess_emb_feat_dim=0,
        sess_emb_hidden_dim=128,
    ):
        super().__init__()
        self.n_parcels = n_parcels
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.loss_start = loss_start
        self.sess_emb_feat_dim = sess_emb_feat_dim

        if not sess_emb_feat_dim:
            raise ValueError(
                "sess_emb_feat_dim must be > 0 (precomputed-feature conditioning)."
            )

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.channel_emb = nn.Embedding(n_parcels, d_model)
        self.sess_emb_mlp = nn.Sequential(
            nn.Linear(sess_emb_feat_dim, sess_emb_hidden_dim),
            nn.GELU(),
            nn.Linear(sess_emb_hidden_dim, sess_emb_hidden_dim),
            nn.GELU(),
            nn.Linear(sess_emb_hidden_dim, d_model),
        )

        self.register_buffer("session_features", torch.zeros(0), persistent=False)

        # Backbone: Mamba-3 (MIMO) + SwiGLU FFN, pre-norm residual
        self.mamba_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba3(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    is_mimo=True,
                    mimo_rank=mimo_rank,
                    is_outproj_norm=is_outproj_norm,
                    chunk_size=8,  # for MIMO kernel, see README "Kernel constraints"
                    layer_idx=i,
                    n_layer=n_layers,
                )
                for i in range(n_layers)
            ]
        )
        self.ffn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        d_ff = ((4 * d_model // 3) + 7) // 8 * 8
        self.ffns = nn.ModuleList(
            [SwiGLUFFN(d_model, d_ff=d_ff) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(d_model)

        # Output head
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        # GPT-2-style residual scaling: each layer contributes 2 residual adds
        # (Mamba + FFN), so scale every residual-branch exit by 1/sqrt(2·N) to
        # keep the residual-stream variance bounded at init across depth. The
        # token/channel/session embeddings init std 0.02 (added at the input).
        assert hasattr(self.mamba_blocks[0], "out_proj"), (
            "Mamba3 has no `out_proj` — library API changed; residual init "
            "scaling would silently be a no-op."
        )
        scale = 1.0 / math.sqrt(2.0 * self.n_layers)

        with torch.no_grad():
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.channel_emb.weight, mean=0.0, std=0.02)

            # Start the embedding MLP near zero (mirrors the std=0.02 embedding init)
            # so the backbone warms up before the session term kicks in.
            nn.init.normal_(self.sess_emb_mlp[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.sess_emb_mlp[-1].bias)
            for ffn in self.ffns:
                ffn.w_out.weight.mul_(scale)
            for m in self.mamba_blocks:
                m.out_proj.weight.mul_(scale)

            # Small head init so initial predictions are near-uniform (CE ≈ logV).
            self.head.weight.mul_(scale)
            self.head.bias.zero_()

    def set_session_features(self, table):
        """Install the per-(session, parcel) feature table for feature conditioning.

        table: (n_sessions, n_parcels, sess_emb_feat_dim) float — indexed by
        (session_ids, channel_ids) in forward(). Stored as a non-persistent buffer
        (kept out of the checkpoint; regenerate with compute_session_features.py).
        """
        dev = next(self.parameters()).device
        self.session_features = table.to(dev, dtype=torch.float32)

    @classmethod
    def from_pretrained(cls, model_dir, allow_new=(), device=None, **overrides):
        """Build from <model_dir>/model_config.json and load <model_dir>/weights.pt.

        Parameters introduced by a subclass (e.g. a conditioning adapter) are allowed
        only if their qualified name contains one of ``allow_new``; any other missing
        or unexpected key is a hard error.
        ``overrides`` supply extra __init__ args a subclass needs (e.g. n_stim). The
        loaded config dict is stashed on the model as ``.pretrained_config``.
        """
        cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
        model = cls(**{**cfg, **overrides})
        state = torch.load(
            os.path.join(model_dir, "weights.pt"), map_location="cpu", weights_only=True
        )
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not any(a in k for a in allow_new)]
        assert not bad, f"missing non-adapter keys: {bad}"
        assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
        model.pretrained_config = cfg
        return model.to(device) if device is not None else model

    def embed(self, tokens, channel_ids, session_ids, **cond):
        """Token embedding + additive channel/session conditioning → (N, L, d_model).

        Subclasses override to inject extra additive INPUT conditioning (e.g. a
        time-varying stimulus term). Unknown ``**cond`` kwargs are ignored here so
        forward() can thread them through uniformly.
        """
        if self.session_features.numel() == 0:
            raise RuntimeError(
                "session feature table is empty — call set_session_features(table) "
                "after constructing/loading the model (it is a non-persistent "
                "buffer, so it is not restored from the checkpoint)."
            )
        h = self.token_emb(tokens)  # (N, L, d_model)
        # Additive channel + feature-derived session conditioning, broadcast over L.
        h = h + self.channel_emb(channel_ids).unsqueeze(1)
        feats = self.session_features[
            session_ids, channel_ids
        ]  # (N, sess_emb_feat_dim)
        h = h + self.sess_emb_mlp(feats).unsqueeze(1)
        return h

    def backbone(self, h, inference_params=None):
        """Mamba-3 (MIMO) + SwiGLU FFN stack, pre-norm residual → (N, L, d_model)."""
        for m_norm, mamba, f_norm, ffn in zip(
            self.mamba_norms, self.mamba_blocks, self.ffn_norms, self.ffns
        ):
            h = h + mamba(m_norm(h), inference_params=inference_params)
            h = h + ffn(f_norm(h))
        return h

    def readout(self, h, **cond):
        """final RMSNorm → shared head → (N, L, V).

        Subclasses override to inject OUTPUT-side conditioning before the head
        (kernel-safe; no gradient through the scan). ``**cond`` ignored here.
        """
        return self.head(self.final_norm(h))

    def forward(
        self,
        tokens,
        channel_ids,
        session_ids,
        inference_params=None,
        **cond,
    ):
        """
        Args:
            tokens: (N, L) int tokens in [0, vocab_size) — N channel-sequences.
            channel_ids: (N,) int parcel index for each sequence.
            session_ids: (N,) int session index for each sequence; indexes the
                per-(session,parcel) feature table (session_features[sid, cid]).
            inference_params: optional mamba_ssm InferenceParams for O(1)-per-step
                AR generation (each row caches its own conv/SSM state).
            **cond: optional extra conditioning inputs, threaded to embed()/readout()
                (base model ignores them; subclasses consume them).

        Returns:
            logits: (N, L, V) next-step token logits.
        """
        h = self.embed(tokens, channel_ids, session_ids, **cond)
        h = self.backbone(h, inference_params=inference_params)
        return self.readout(h, **cond)

    def loss_from_output(self, logits, target_tokens):
        """Cross-entropy next-token loss.

        The first ``loss_start`` positions are excluded so the SSM state can
        warm up from its zero initial condition.

        Args:
            logits:        (N, L, V) from ``forward``.
            target_tokens: (N, L) int64 next-step targets.

        Returns:
            (loss, metrics) — scalar loss and a dict of float metrics.
        """
        w = self.loss_start
        if target_tokens.shape[1] <= w:
            raise ValueError(
                f"seq_len ({target_tokens.shape[1]}) must be > loss_start "
                f"({w}); otherwise the loss slice is empty."
            )

        logits = logits[:, w:]  # (N, L', V)
        target = target_tokens[:, w:]  # (N, L')

        V = logits.shape[-1]
        flat_logits = logits.reshape(-1, V)
        flat_target = target.reshape(-1).long()

        loss = F.cross_entropy(flat_logits, flat_target, reduction="mean")

        with torch.no_grad():
            log_p = F.log_softmax(flat_logits.float(), dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1).mean()
            top1_prob = p.max(dim=-1).values.mean()
            preds = logits.argmax(dim=-1)  # (N, L')
            top1_acc = (preds == target).float().mean().item()
            ppl = loss.exp().item()

        metrics = {
            "ce": loss.item(),
            "perplexity": ppl,
            "top1_acc": top1_acc,
            "entropy": entropy.item(),
            "top1_prob": top1_prob.item(),
        }

        return loss, metrics
