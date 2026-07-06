"""
Exp 08 smoke: 动态 σ_eff 高斯投影 + 尺度调制 FNO
====================================================

动机 (cwf-manifesto, 紧接 exp07 0/6 窗口 DEAD 之后):
  exp05 已经证 Stage A 重建可 PASS (complex 0.542), Stage B 多次证伪.
  用户最新提案引入了 2 个新点, 之前 exp05/exp07 都没测:
    (1) 动态高斯核 σ_eff = α/L (而非固定 σ 或 CNN 自学习)
    (2) FNO 谱权重用 norm_L 注入尺度调制 (而非静态谱权重)

  本探针隔离这 2 个新点, 不重新发明 exp05:
    Smoke A: Gaussian encoder (动态 σ) + 简单复数 decoder → 测 (1)
    Smoke B: 复用 exp05 Stage A ckpt + ScaleModulatedFNO 替代静态 FNO → 测 (2)

判决 (相对 exp05 baseline, 单 seed 1000 步):
  Smoke A: exp05 Stage A complex @1000 = 1.222
    < 1.222 → 动态 σ 改善, 值得继续
    ~ 1.222 → 动态 σ 不优于 CNN 自学习, 归档
    > 1.222 → 动态 σ 有害, 立即归档
  Smoke B: exp05 Stage B complex @1000 = 2.876
    < 2.876 → 尺度调制改善
    ~ 2.876 → 调制无意义
    > 2.876 → 调制无帮助, 归档

硬纪律:
  - 复用 exp05 data loader + 训练循环 (避免重写 100+ 行)
  - 1 个新文件, 2 个新模块, 总新代码 < 60 行
  - 不复现 exp05 已证内容 (Stage A CNN / Stage B 静态 FNO)
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
VAL_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UNIFORM_LOSS = math.log(256)


# ============================================================
# 复用 exp05 工具 (data load, 复数 modReLU, 训练循环)
# ============================================================
def load_data(n_train=2_000_000, n_val=100_000):
    tr = "\n".join(pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).tolist()).encode("utf-8")[:n_train]
    va = "\n".join(pd.read_parquet(VAL_PARQUET)["text"].astype(str).tolist()).encode("utf-8")[:n_val]
    return torch.tensor(list(tr), dtype=torch.int64), torch.tensor(list(va), dtype=torch.int64)

def get_batch(ids, bs, sl):
    n = len(ids) - sl - 1
    s = torch.randint(0, n, (bs,))
    return torch.stack([ids[i:i + sl] for i in s]).to(DEVICE)

def get_batch_with_lengths(ids, bs, sl):
    """返回 (x, lengths), 用真实长度 (sl) — Gaussian encoder 需要."""
    x = get_batch(ids, bs, sl)
    return x, torch.full((bs,), sl, dtype=torch.long, device=DEVICE)

def get_batch_next(ids, bs, sl):
    n = len(ids) - sl - 1
    s = torch.randint(0, n, (bs,))
    return (torch.stack([ids[i:i + sl] for i in s]).to(DEVICE),
            torch.stack([ids[i + sl] for i in s]).to(DEVICE))

def cmodrelu(z):
    m = torch.abs(z)
    return torch.tanh(m) * (z / torch.clamp(m, min=1e-8))


# ============================================================
# 新想法 (1): 动态 σ_eff = α/L 高斯场编码器
# 替代 exp05 的 CNN encoder. 无卷积, 直接 Gaussian projection.
# ============================================================
class DynamicGaussianFieldEncoder(nn.Module):
    """Byte seq → 复数波场 Ψ_0 ∈ C^{B, M, d}, σ_eff = α/L 抗混叠.
    关键点 (vs exp05):
      - 相对坐标 x_i = i/L ∈ (0, 1]
      - σ_eff ∝ 1/L (自适应, 长文本不混叠)
      - 满载 M 网格, 无 Mask/Padding 阶跃
    """
    def __init__(self, vocab_size, d, M, alpha=1.0):
        super().__init__()
        self.M, self.alpha, self.d = M, alpha, d
        self.er = nn.Embedding(vocab_size, d)
        self.ei = nn.Embedding(vocab_size, d)
        # 网格坐标 x_m ∈ (0, 1], M 个点
        self.register_buffer('grid_x', torch.linspace(1 / M, 1.0, M))

    def forward(self, byte_ids, lengths):
        B, L = byte_ids.shape
        e = torch.complex(self.er(byte_ids), self.ei(byte_ids))  # (B, L, d)
        pos = torch.arange(1, L + 1, device=byte_ids.device).float()
        x_i = pos.unsqueeze(0) / lengths.float().unsqueeze(1)  # (B, L) ∈ (0, 1]
        mask = (pos.unsqueeze(0) <= lengths.unsqueeze(1)).float()
        x_i = x_i * mask
        sigma = (self.alpha / lengths.float()).clamp(min=1e-4).view(B, 1, 1)
        d2 = (self.grid_x.view(1, -1, 1) - x_i.unsqueeze(1)) ** 2  # (B, M, L)
        w = torch.exp(-d2 / (2 * sigma ** 2)) * mask.unsqueeze(1)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        # w 是实数权重, e 是 cfloat: 分别对实部/虚部做 matmul
        return torch.complex(torch.matmul(w, e.real), torch.matmul(w, e.imag))


# ============================================================
# Smoke A 模型: Gaussian encoder + 简单复数 decoder
# ============================================================
class SmokeA(nn.Module):
    """Smoke A: 动态高斯场 + 简单复数 decoder. Stage A 重构 only.
    测的是新想法 (1): 动态 σ_eff vs CNN 自学习 encoder."""
    def __init__(self, vocab_size=256, d=32, M=64, alpha=1.0, seq_len=256):
        super().__init__()
        self.enc = DynamicGaussianFieldEncoder(vocab_size, d, M, alpha)
        # 简单 decoder: M → L 上采样 + 复数 conv + head
        self.dec_conv1 = nn.Conv1d(d, d, 5, padding=2)  # 复数走实/虚两组
        self.dec_conv2 = nn.Conv1d(d, d, 5, padding=2)
        self.head = nn.Linear(2 * d, vocab_size)
        self.M, self.seq_len = M, seq_len

    def forward(self, x, lengths):
        psi = self.enc(x, lengths)  # (B, M, d) cfloat
        z = psi.permute(0, 2, 1)  # (B, d, M)
        # 复数 decoder: 实/虚分别 interpolate + 实 conv
        target = self.seq_len
        zr = F.interpolate(z.real, size=target, mode="linear", align_corners=False)
        zi = F.interpolate(z.imag, size=target, mode="linear", align_corners=False)
        zr = F.gelu(self.dec_conv1(zr))
        zi = F.gelu(self.dec_conv1(zi))
        zr = F.gelu(self.dec_conv2(zr))
        zi = F.gelu(self.dec_conv2(zi))
        flat = torch.cat([zr, zi], dim=1).permute(0, 2, 1)  # (B, L, 2d)
        return self.head(flat)


# ============================================================
# 新想法 (2): 尺度调制 FNO 谱卷积
# norm_L → MLP → 复数调制因子 (per-mode) → W_modulated = W * modulator
# ============================================================
class ScaleModulatedSpectralConv1d(nn.Module):
    """在 exp04/exp05 ComplexSpectralConv1d 基础上加 norm_L 注入."""
    def __init__(self, channels, modes):
        super().__init__()
        self.modes, self.channels = modes, channels
        self.weights = nn.Parameter(
            torch.randn(channels, channels, modes, dtype=torch.cfloat) / (channels * modes))
        # 调制 MLP: 1 (norm_L) → hidden → 2*modes (实+虚)
        self.mod = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 2 * modes))

    def forward(self, x, norm_L):
        """x: (B, C, L) cfloat.  norm_L: (B, 1) real = L/M."""
        B, C, L = x.shape
        x_ft = torch.fft.fft(x, dim=-1)
        eff = min(self.modes, L // 2)
        # 生成 per-mode 复数调制 (B, eff)
        mr, mi = self.mod(norm_L).chunk(2, dim=-1)
        modulator = torch.complex(mr, mi)[:, :eff]  # (B, eff)
        # 调制: W_mod[b, o, i, m] = W[o, i, m] * modulator[b, m]
        W = self.weights[:, :, :eff]  # (C, C, eff)
        W_mod = W.unsqueeze(0) * modulator.view(B, 1, 1, eff)  # (B, C, C, eff)
        out_ft = torch.einsum('bim,boim->bom', x_ft[:, :, :eff], W_mod)
        full = torch.zeros(B, C, L, dtype=x_ft.dtype, device=x.device)
        full[:, :, :eff] = out_ft
        return torch.fft.ifft(full, dim=-1)


# ============================================================
# Smoke B 模型: 加载 exp05 Stage A ckpt + ScaleModulated FNO
# ============================================================
class ScaleModulatedFNOBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.spec = ScaleModulatedSpectralConv1d(channels, modes)
        self.local_r = nn.Conv1d(channels, channels, 1)
        self.local_i = nn.Conv1d(channels, channels, 1)

    def forward(self, x, norm_L):
        h = self.spec(x, norm_L) + torch.complex(self.local_r(x.real), self.local_i(x.imag))
        h = cmodrelu(h)
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


class SmokeB(nn.Module):
    """Smoke B: 复用 exp05 冻结 encoder, 替换 FNO 为尺度调制版.
    测的是新想法 (2): 尺度调制 vs 静态 FNO."""
    def __init__(self, d=32, M=64, modes=16, n_layers=2, seq_len=256):
        super().__init__()
        self.blocks = nn.ModuleList([ScaleModulatedFNOBlock(d, modes) for _ in range(n_layers)])
        self.next_head = nn.Linear(2 * d, 256)
        self.M, self.seq_len = M, seq_len

    def forward(self, psi_0, norm_L):
        # psi_0: (B, M, d) cfloat 来自冻结 exp05 encoder
        z = psi_0.permute(0, 2, 1)  # (B, d, M)
        norm_L_3d = norm_L  # (B, 1) -> broadcast 在 B 维
        for blk in self.blocks:
            z = blk(z, norm_L_3d)
        last = z[:, :, -1]  # (B, d) 末端
        return self.next_head(torch.cat([last.real, last.imag], dim=-1))


# ============================================================
# 训练循环 (复用 exp05 模式)
# ============================================================
def train_smoke_a(steps, d, M, alpha, batch_size, peak_lr, seed, eval_steps):
    """Smoke A: Gaussian encoder 重建. 对照 exp05 Stage A."""
    torch.manual_seed(seed)
    print(f"\n{'='*70}\n[SmokeA] dynamic σ encoder, α={alpha}, M={M}, d={d}\n{'='*70}")
    train_ids, val_ids = load_data()
    model = SmokeA(d=d, M=M, alpha=alpha).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))
    seq_len = model.seq_len
    trace = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = peak_lr * min(step / 100, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x, lens = get_batch_with_lengths(train_ids, batch_size, seq_len)
        logits = model(x, lens)
        loss = F.cross_entropy(logits.reshape(-1, 256), x.reshape(-1))
        if torch.isnan(loss):
            print("  NaN, abort"); return None
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in eval_steps or step == steps:
            model.eval()
            with torch.no_grad():
                vtotal, vcnt = 0.0, 0
                for _ in range(20):
                    x, lens = get_batch_with_lengths(val_ids, 8, seq_len)
                    vtotal += F.cross_entropy(model(x, lens).reshape(-1, 256), x.reshape(-1)).item() * x.numel()
                    vcnt += x.numel()
            vl = vtotal / vcnt
            trace.append({"step": step, "val": round(vl, 4), "train": round(loss.item(), 4)})
            print(f"  step {step:>4}  train={loss.item():.4f}  val={vl:.4f}  "
                  f"(uniform={UNIFORM_LOSS:.4f})  t={time.time()-t0:.0f}s", flush=True)
            model.train()
    out = RESULTS_DIR / f"smoke_a_α{alpha}_M{M}_d{d}_s{seed}.json"
    out.write_text(json.dumps({"trace": trace, "uniform": UNIFORM_LOSS,
                                "exp05_baseline_complex_d32_@1000": 1.222}, indent=2))
    print(f"[saved] -> {out}")
    return trace


def train_smoke_b(steps, d, M, modes, n_layers, batch_size, peak_lr, seed, eval_steps):
    """Smoke B: exp05 ckpt + ScaleModulated FNO. 对照 exp05 Stage B @1000=2.876."""
    torch.manual_seed(seed)
    print(f"\n{'='*70}\n[SmokeB] scale-modulated FNO, d={d}, M={M}, modes={modes}\n{'='*70}")
    train_ids, val_ids = load_data()
    # 加载 exp05 Stage A 冻结 encoder
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wave_autoencoder import CWFModelComplex
    full = CWFModelComplex(d=d, modes=modes, n_layers=n_layers, seq_len=256, stride=4).to(DEVICE)
    ckpt = RESULTS_DIR / "tokenizer_complex.pt"
    if not ckpt.exists():
        ckpt = Path(__file__).resolve().parent / "results" / "tokenizer_complex.pt"
    sd = torch.load(ckpt, map_location=DEVICE)
    full.load_state_dict(sd, strict=False)
    print(f"[load] exp05 tokenizer ckpt -> {ckpt}")
    for p in full.tokenizer.parameters():
        p.requires_grad = False
    # 替换 FNO + head
    smoke_b = SmokeB(d=d, M=M, modes=modes, n_layers=n_layers).to(DEVICE)
    opt = torch.optim.AdamW(smoke_b.parameters(), lr=peak_lr, weight_decay=0.01, betas=(0.9, 0.95))
    seq_len = 256
    trace = []
    t0 = time.time()
    for step in range(1, steps + 1):
        lr = peak_lr * min(step / 100, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch_next(train_ids, batch_size, seq_len)
        with torch.no_grad():
            psi_0 = full.tokenizer.encode(x)  # (B, M, d) cfloat
        norm_L = (torch.tensor([seq_len], device=DEVICE).float() / M).expand(x.size(0), 1)
        logits = smoke_b(psi_0, norm_L)  # (B, 256)
        loss = F.cross_entropy(logits, y)
        if torch.isnan(loss):
            print("  NaN, abort"); return None
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(smoke_b.parameters(), 1.0); opt.step()
        if step in eval_steps or step == steps:
            smoke_b.eval()
            with torch.no_grad():
                vtotal, vcnt = 0.0, 0
                for _ in range(40):
                    x, y = get_batch_next(val_ids, 8, seq_len)
                    psi_0 = full.tokenizer.encode(x)
                    norm_L = (torch.tensor([seq_len], device=DEVICE).float() / M).expand(x.size(0), 1)
                    logits = smoke_b(psi_0, norm_L)
                    vtotal += F.cross_entropy(logits, y).item() * y.numel()
                    vcnt += y.numel()
            vl = vtotal / vcnt
            trace.append({"step": step, "val": round(vl, 4), "train": round(loss.item(), 4)})
            print(f"  step {step:>4}  train={loss.item():.4f}  val={vl:.4f}  "
                  f"(uniform={UNIFORM_LOSS:.4f})  t={time.time()-t0:.0f}s", flush=True)
            smoke_b.train()
    out = RESULTS_DIR / f"smoke_b_M{M}_modes{modes}_d{d}_s{seed}.json"
    out.write_text(json.dumps({"trace": trace, "uniform": UNIFORM_LOSS,
                                "exp05_baseline_complex_@1000": 2.876}, indent=2))
    print(f"[saved] -> {out}")
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", choices=["a", "b", "both"], default="both")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--peak_lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_steps", nargs="+", type=int, default=[200, 500, 1000])
    p.add_argument("--alpha", type=float, default=1.0, help="动态 σ 系数: σ_eff = α/L")
    p.add_argument("--d_model", type=int, default=32)
    p.add_argument("--M", type=int, default=64, help="网格大小 M=seq_len/stride=256/4")
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--n_layers", type=int, default=2)
    args = p.parse_args()

    if args.ablation in ("a", "both"):
        train_smoke_a(args.steps, args.d_model, args.M, args.alpha,
                       args.batch_size, args.peak_lr, args.seed, args.eval_steps)
    if args.ablation in ("b", "both"):
        train_smoke_b(args.steps, args.d_model, args.M, args.modes, args.n_layers,
                       args.batch_size, args.peak_lr, args.seed, args.eval_steps)


if __name__ == "__main__":
    main()
