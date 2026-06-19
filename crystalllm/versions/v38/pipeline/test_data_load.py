"""
v38 数据加载 sanity test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from z_health_check import load_val_data, encode_val_with_encoder, encode_text_for_mi, load_v24_encoder

print("--- Loading val data ---")
texts, z = load_val_data()
print(f"texts: {len(texts)}, z: {z.shape}")
print(f"first text: {texts[0][:80]}")
print(f"z mean: {z.mean():.4f}, std: {z.std():.4f}")
print(f"z[0, :5]: {z[0, :5]}")

print("\n--- Loading v24 encoder ---")
encoder_ckpt = load_v24_encoder("cpu")
print(f"encoder_ckpt keys: {list(encoder_ckpt.keys())}")
if "config" in encoder_ckpt:
    print(f"config: {encoder_ckpt['config']}")

print("\n--- Encoding val (cache) ---")
z_tensor = encode_val_with_encoder(encoder_ckpt, "cpu")
print(f"z_tensor: {z_tensor.shape}, dtype={z_tensor.dtype}")

print("\n--- Encoding text features (for MINE) ---")
emb = encode_text_for_mi(texts, "cpu")
print(f"emb shape: {emb.shape}, dtype={emb.dtype}")
print(f"emb[0]: {emb[0]}")

print("\nAll data loading tests passed")