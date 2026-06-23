"""Smoke test: 验证 Exp 29 能加载 v49 1.2B checkpoint 并跑一次 forward."""
import torch, sys, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.v49_pre.exp_runner import Transformer50M

print("[smoke] loading 1.2B checkpoint ...")
ckpt = torch.load(PROJECT_ROOT / "experiments/v49_pre/results/v49_scale_1.2b.final.pt", map_location="cpu", weights_only=False)
args = ckpt['args']
print(f"[smoke] args: {args}")
print(f"[smoke] train val_ppl: {ckpt['val_ppl']:.4f}")

model = Transformer50M(
    vocab_size=2261,
    d_model=args['d_model'], n_layers=args['n_layers'],
    n_heads=args['n_heads'], d_ff=args['d_ff'],
    max_seq_len=512, dropout=0.0,
)
model.load_state_dict(ckpt['model_state'])
n_params = sum(p.numel() for p in model.parameters())
print(f"[smoke] params: {n_params:,} ({n_params/1e9:.2f}B)")

print("[smoke] moving to GPU ...")
model = model.to("cuda")
model.eval()
print(f"[smoke] GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("[smoke] testing forward pass ...")
x = torch.randint(0, 100, (1, 32), device="cuda")
with torch.no_grad():
    logits = model(x)
print(f"[smoke] logits shape: {logits.shape}")
print(f"[smoke] GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB / peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("[smoke] OK")
