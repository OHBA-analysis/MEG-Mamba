# MEG-Mamba

### Architecture

```
Inputs   tokens (N, L)    channel_ids (N,)    session_ids (N,)
         N channel-sequences — one parcel's token stream each

Embed                                                 → (N, L, d_model)
         token_emb[tokens]                            token lookup
         + channel_emb[channel_ids]                   per-parcel lookup
         + sess_emb_mlp(session_features[sid, cid])   zero-shot session MLP

Backbone   × N_layers, pre-norm residual              → (N, L, d_model)
           h = h + Mamba3-MIMO( RMSNorm(h) )          complex-valued SSM (mimo_rank)
           h = h + SwiGLU-FFN(  RMSNorm(h) )          gated FFN, d_ff ≈ 4·d_model/3

Head       logits = Linear( RMSNorm(h) )              → (N, L, V)
           loss   = next token cross-entropy
```

### Hyperparameters

| Setting | Value |
|---|---|
| n_parcels / vocab | 100 (Schaefer100) / 92 |
| d_model / n_layers | 256 / 4 |
| parameters | 3.40 M |
| d_state / headdim / expand | 64 / 64 / 2 |
| mimo_rank | 4  (kernel requires ≥4) |
| seq_len / loss_start | 1000 (4 s) / 50 |
| channel emb | per-parcel lookup, added to token-emb at input |
| session emb | MLP over precomputed unigram + bigram-PCA features, added to token-emb (zero-shot) |
| session features | unigram (92) + bigram-PCA (200) per (session, parcel); sqrt → train-only standardise → PCA |
| batch_size | channel-sequences/step (= kernel batch rows); 512 |
| precision | bf16 AMP |
| optimizer | AdamW (0.9, 0.95), weight_decay 0.01, grad_clip 1.0 |
| LR schedule | warmup 2000 steps → constant 1e-4 |
| training | 3 epochs, ~22 h on 1× NVIDIA A100 (80 GB) |
