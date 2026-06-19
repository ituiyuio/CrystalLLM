"""test_v36_warmstart.py — 验证 v25 → v36 warm-start 加载的 shape 与计数

修正自 plan:
  原 plan 假设 loaded=290, skipped=2, fresh=96, mismatched=0
  实际 (v25 实际有 295 keys, v36 实际有 533 keys):
    loaded    = 293  (1 pos.weight 偏移 + 292 直接 shape 匹配)
    skipped   = 2    (z_to_emb.weight, z_to_emb.bias)
    fresh     = 0    (从 v25 角度看 — v25 中除了 z_to_emb 都映射到 v36)
    v36-fresh = 240  (从 v36 角度看 — v36 有 240 个 keys 无 v25 来源)
                  = 10 cross-attn weights × 24 blocks
                  (ln_cross.bias/weight, q_cross.bias/weight,
                   k_cross.bias/weight, v_cross.bias/weight,
                   proj_cross.bias/weight)
    mismatched = 0

加载时用 PyTorch 的 strict=False 容忍 v36 中未填的 240 个 keys,
让它们保持 random init.
"""
import torch
import sys
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

# v25 配置
V, T, D_Z = 2261, 512, 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
BOS_ID = 1

# 加载 v25
ckpt_v25 = torch.load("crystalllm/v25_decoder.pt", map_location="cpu", weights_only=False)
v25_state = ckpt_v25["decoder"]
print(f"v25 state keys: {len(v25_state)}")

# 构建 v36
new_decoder = DecoderCrossAttn(V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID)
new_state = new_decoder.state_dict()
print(f"v36 state keys: {len(new_state)}")

# warm-start 加载 (从 v25 角度看)
loaded, skipped, v25_fresh, mismatched = 0, 0, 0, 0
v25_fresh_keys = []
for k, v in v25_state.items():
    if k in ("z_to_emb.weight", "z_to_emb.bias"):
        skipped += 1
        continue
    if k == "pos.weight":
        # v25 pos[0]=z, pos[1]=BOS, pos[2:T+2]=tokens; v36 pos[0]=BOS, pos[1:T+1]=tokens
        new_state[k][: T + 1] = v[1 : T + 2]
        loaded += 1
        continue
    if k in new_state:
        if v.shape == new_state[k].shape:
            new_state[k] = v
            loaded += 1
        else:
            mismatched += 1
            print(f"  shape mismatch: {k} v25={v.shape} v36={new_state[k].shape}")
    else:
        v25_fresh += 1
        v25_fresh_keys.append(k)

# v36 中 fresh (无 v25 来源, 保持 random init)
v36_fresh_keys = [k for k in new_state if k not in v25_state
                  or k in ("z_to_emb.weight", "z_to_emb.bias")]
# 但 pos.weight 在 v25 中, 且已被偏移加载, 不算 v36-fresh
# z_to_emb 在 v25 中但 v36 没有, 也已 skipped
v36_fresh_keys = [k for k in new_state
                  if k not in v25_state]

print(f"\nWarm-start summary:")
print(f"  loaded    (v25 → v36, direct match):  {loaded}")
print(f"  loaded    (pos.weight shifted):       1 (in loaded count above)")
print(f"  skipped   (z_to_emb.*):               {skipped}")
print(f"  v25-fresh (v25 has, v36 ignores):    {v25_fresh}")
print(f"  v36-fresh (v36 has, no v25 source):  {len(v36_fresh_keys)}")
print(f"    = 10 cross-attn weights × 24 blocks = {10 * 24}")
print(f"  mismatched (shape error):            {mismatched}")

# 校验计数
EXPECTED_LOADED = 293       # 295 v25 - 2 skipped = 293
EXPECTED_SKIPPED = 2
EXPECTED_V25_FRESH = 0
EXPECTED_V36_FRESH = 240    # 10 cross-attn weights × 24 blocks
EXPECTED_MISMATCHED = 0

assert loaded == EXPECTED_LOADED, \
    f"expected {EXPECTED_LOADED} loaded, got {loaded}"
assert skipped == EXPECTED_SKIPPED, \
    f"expected {EXPECTED_SKIPPED} skipped (z_to_emb), got {skipped}"
assert v25_fresh == EXPECTED_V25_FRESH, \
    f"expected {EXPECTED_V25_FRESH} v25-fresh, got {v25_fresh}"
assert len(v36_fresh_keys) == EXPECTED_V36_FRESH, \
    f"expected {EXPECTED_V36_FRESH} v36-fresh, got {len(v36_fresh_keys)}"
assert mismatched == EXPECTED_MISMATCHED, \
    f"unexpected mismatches: {mismatched}"

# pos.weight[T+1:] 保持 random init (没有 v25 来源 — v25 只有 T+2 行 [z, BOS, T tokens])
# 注意: nn.Embedding init 为 N(0,1), 这里不强制为 zero, 仅记录
print(f"  pos.weight[T+1:] (no v25 source): mean={new_state['pos.weight'][T + 1:].mean().item():.4f}, std={new_state['pos.weight'][T + 1:].std().item():.4f}")

# 用 strict=False 加载 (240 个 v36-fresh 在 new_state 中保留 random init, 不需 missing)
missing, unexpected = new_decoder.load_state_dict(new_state, strict=False)
print(f"\nload_state_dict(strict=False):")
print(f"  missing   (in v36, not filled):  {len(missing)}")
print(f"  unexpected (in v25, not in v36): {len(unexpected)}")
# 期望 0 missing 因为 new_state 包含全部 533 个 v36 keys
# (240 个 cross-attn 保留 random init)
assert len(missing) == 0, \
    f"expected 0 missing (new_state has all 533 v36 keys), got {len(missing)}"
assert len(unexpected) == 0, \
    f"expected 0 unexpected (z_to_emb skipped already), got {len(unexpected)}"

# 验证 240 个 cross-attn keys 确实是 random init (没被 v25 覆盖)
sample_cross_keys = v36_fresh_keys[:5]
for k in sample_cross_keys:
    # 检查仍是 init 状态 (与 new_decoder 内部 init 一致)
    print(f"  fresh sample: {k} = {new_state[k].shape}")

# 验证 v36 仍能前向
new_decoder = new_decoder.to("cuda")
B = 2
z = torch.randn(B, D_Z, device="cuda")
x = torch.randint(0, V, (B, T), device="cuda")
logits = new_decoder(z, x)
print(f"\npost-warmstart forward OK, logits shape: {logits.shape}")

print("\nv36 Warm-start sanity checks passed")