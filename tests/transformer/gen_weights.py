#!/usr/bin/env python3
"""
Generate test fixtures for test_transformer.cpp.

Writes two binary files in the current working directory:
  weights.bin  – 7×int32 config header, then all float32 weights in
                 TransformerWeights declaration order
  expected.bin – int32 test-case count, then for each case:
                 int32 token, int32 pos, vocab_size×float32 logits

Run from the CMake build directory; CMakeLists.txt does this automatically.
"""
import math
import struct
import sys
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

DIM        = 8
HIDDEN_DIM = 16
N_LAYERS   = 2
N_HEADS    = 2
N_KV_HEADS = 2
VOCAB_SIZE = 16
SEQ_LEN    = 16

HEAD_SIZE = DIM // N_HEADS               # 4
KV_DIM    = (DIM * N_KV_HEADS) // N_HEADS  # 8

rng = np.random.default_rng(42)

def randf(*shape):
    """Small-scale random float32 weights."""
    return rng.standard_normal(shape).astype(np.float32) * 0.1

# ── Weights ───────────────────────────────────────────────────────────────────

token_embedding_table = randf(VOCAB_SIZE, DIM)
rms_att_weight        = randf(N_LAYERS, DIM)
rms_ffn_weight        = randf(N_LAYERS, DIM)
wq                    = randf(N_LAYERS, DIM, DIM)          # (n_layers, n_heads*hs, dim)
wk                    = randf(N_LAYERS, KV_DIM, DIM)       # (n_layers, n_kv_heads*hs, dim)
wv                    = randf(N_LAYERS, KV_DIM, DIM)
wo                    = randf(N_LAYERS, DIM, DIM)          # (n_layers, dim, dim)
w1                    = randf(N_LAYERS, HIDDEN_DIM, DIM)   # (n_layers, hidden_dim, dim)
w2                    = randf(N_LAYERS, DIM, HIDDEN_DIM)   # (n_layers, dim, hidden_dim)
w3                    = randf(N_LAYERS, HIDDEN_DIM, DIM)
rms_final_weight      = randf(DIM)
wcls                  = randf(VOCAB_SIZE, DIM)

# ── Reference implementation (mirrors C++ exactly, float32 throughout) ────────

def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """C++ rmsnorm: out = weight * x / sqrt(mean(x^2) + 1e-5)"""
    ss = np.float32(np.sum(x * x, dtype=np.float32)) / np.float32(len(x))
    ss = ss + np.float32(1e-5)
    ss = np.float32(1.0) / np.sqrt(ss)
    return (weight * ss * x).astype(np.float32)

def silu(x: np.ndarray) -> np.ndarray:
    return (x * (np.float32(1.0) / (np.float32(1.0) + np.exp(-x)))).astype(np.float32)

def apply_rope(q: np.ndarray, k: np.ndarray, pos: int) -> tuple:
    """In-place RoPE matching C++ attention.hpp exactly."""
    q = q.copy().astype(np.float32)
    k = k.copy().astype(np.float32)
    for i in range(0, DIM, 2):
        head_dim = i % HEAD_SIZE
        # Match: powf(10000.0f, head_dim / (float32_t)head_size)
        freq = np.float32(1.0) / np.float32(10000.0 ** (head_dim / HEAD_SIZE))
        val  = np.float32(pos) * freq
        fcr  = np.float32(math.cos(float(val)))
        fci  = np.float32(math.sin(float(val)))
        rotn = 2 if i < KV_DIM else 1
        for v_idx in range(rotn):
            vec = q if v_idx == 0 else k
            v0, v1 = np.float32(vec[i]), np.float32(vec[i + 1])
            vec[i]     = v0 * fcr - v1 * fci
            vec[i + 1] = v0 * fci + v1 * fcr
    return q, k

def attention_forward(x, wq_l, wk_l, wv_l, key_cache, val_cache, pos):
    """Matches C++ Attention::forward. key_cache/val_cache are flat float32 arrays."""
    q = (wq_l @ x).astype(np.float32)   # (DIM,)
    k = (wk_l @ x).astype(np.float32)   # (KV_DIM,)
    v = (wv_l @ x).astype(np.float32)   # (KV_DIM,)

    # Write raw k,v into the flat cache (C++ writes before RoPE, then rotates in-place)
    key_cache[pos * KV_DIM:(pos + 1) * KV_DIM] = k
    val_cache[pos * KV_DIM:(pos + 1) * KV_DIM] = v

    # RoPE on q and the cached k slice (C++ rotates the cache view in-place)
    k_slice = key_cache[pos * KV_DIM:(pos + 1) * KV_DIM].copy()
    q, k_rotated = apply_rope(q, k_slice, pos)
    key_cache[pos * KV_DIM:(pos + 1) * KV_DIM] = k_rotated

    kv_mul = N_HEADS // N_KV_HEADS
    xb = np.zeros(DIM, dtype=np.float32)

    for h in range(N_HEADS):
        q_h  = q[h * HEAD_SIZE:(h + 1) * HEAD_SIZE]
        att  = np.zeros(pos + 1, dtype=np.float32)

        for t in range(pos + 1):
            base = t * KV_DIM + (h // kv_mul) * HEAD_SIZE
            k_t  = key_cache[base:base + HEAD_SIZE]
            score = np.float32(np.dot(q_h, k_t)) / np.float32(math.sqrt(HEAD_SIZE))
            att[t] = score

        # numerically-stable softmax (matches C++ softmax template)
        att = att - att.max()
        att = np.exp(att).astype(np.float32)
        att = (att / att.sum()).astype(np.float32)

        xb_h = np.zeros(HEAD_SIZE, dtype=np.float32)
        for t in range(pos + 1):
            base = t * KV_DIM + (h // kv_mul) * HEAD_SIZE
            v_t  = val_cache[base:base + HEAD_SIZE]
            xb_h += np.float32(att[t]) * v_t
        xb[h * HEAD_SIZE:(h + 1) * HEAD_SIZE] = xb_h

    return xb.astype(np.float32)

def transformer_forward(token, pos, key_caches, val_caches):
    """Full transformer forward matching C++ Transformer::forward."""
    x = token_embedding_table[token].copy().astype(np.float32)

    for layer in range(N_LAYERS):
        # Attention sub-layer
        xh  = rmsnorm(x, rms_att_weight[layer])
        xh  = attention_forward(xh, wq[layer], wk[layer], wv[layer],
                                key_caches[layer], val_caches[layer], pos)
        xh2 = (wo[layer] @ xh).astype(np.float32)
        x   = (x + xh2).astype(np.float32)

        # FFN sub-layer
        xh  = rmsnorm(x, rms_ffn_weight[layer])
        hb  = (w1[layer] @ xh).astype(np.float32)
        hb2 = (w3[layer] @ xh).astype(np.float32)
        hb  = (silu(hb) * hb2).astype(np.float32)
        xh  = (w2[layer] @ hb).astype(np.float32)
        x   = (x + xh).astype(np.float32)

    x      = rmsnorm(x, rms_final_weight)
    logits = (wcls @ x).astype(np.float32)
    return logits

# ── Generate test cases ───────────────────────────────────────────────────────

key_caches = [np.zeros(SEQ_LEN * KV_DIM, dtype=np.float32) for _ in range(N_LAYERS)]
val_caches = [np.zeros(SEQ_LEN * KV_DIM, dtype=np.float32) for _ in range(N_LAYERS)]

test_cases = []

# pos=0, token=0
logits0 = transformer_forward(0, 0, key_caches, val_caches)
test_cases.append((0, 0, logits0))

# pos=1, token=3 — shares KV caches with the previous call
logits1 = transformer_forward(3, 1, key_caches, val_caches)
test_cases.append((3, 1, logits1))

# ── Write binary files ────────────────────────────────────────────────────────

with open("weights.bin", "wb") as f:
    f.write(struct.pack("7i", DIM, HIDDEN_DIM, N_LAYERS, N_HEADS,
                        N_KV_HEADS, VOCAB_SIZE, SEQ_LEN))
    for arr in [token_embedding_table, rms_att_weight, rms_ffn_weight,
                wq, wk, wv, wo, w1, w2, w3, rms_final_weight, wcls]:
        f.write(arr.astype(np.float32).tobytes())

with open("expected.bin", "wb") as f:
    f.write(struct.pack("i", len(test_cases)))
    for token, pos, logits in test_cases:
        f.write(struct.pack("2i", token, pos))
        f.write(logits.astype(np.float32).tobytes())

print(f"Wrote {len(test_cases)} test cases")
print(f"  weights.bin: dim={DIM} hidden={HIDDEN_DIM} layers={N_LAYERS} "
      f"heads={N_HEADS} vocab={VOCAB_SIZE} seq={SEQ_LEN}")
for i, (tok, p, lg) in enumerate(test_cases):
    print(f"  case {i}: token={tok} pos={p}  "
          f"logits[0:4]={lg[:4].tolist()}")
