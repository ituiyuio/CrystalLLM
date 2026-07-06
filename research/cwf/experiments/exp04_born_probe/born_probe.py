"""
Exp 04: Born Probe — 复数波场表征在 byte-level 文本上是否可优化?
=================================================================

动机 (cwf-manifesto 分支, 路线决策):
  CWF 宣言的 Phase 2 (Lorenz) 尚未越过 GO gate (EPT ~4.6 vs 要求 >=30).
  直接跳到 Phase 4 (自然语言) 做全盘设计是工程大忌. 本探针剥离所有宏大
  叙事, 只问一个最根本的问题:

      在 byte-level 离散文本上, 复数波场 + Born 法则的梯度, 到底能不能优化?

  不追求 SOTA, 不追求闭包结构, 不加 L_energy / L_spec. 纯粹验证:
  (1) 收敛性 (loss 能否显著低于均匀分布 ln(256)=5.545)
  (2) 梯度稳定性 (复数 Wirtinger 梯度在 FP32 是否 NaN / 消失)

设计 (3 处提炼, 见对话记录):
  提炼1 - 读取 M 位置: 在最后一个网格点读取 psi_pred (bigram 级别学习也算 GO).
          不做 M+1 的零 query slot (避免 257 点 FFT, 257 是素数对 FFT 不友好).
  提炼2 - sigma 扫描: sigma ∈ {0, 0.5, 1.0}. sigma=0 是 delta 对照, 锁定变量.
          若 sigma=0 收敛而 sigma>0 失败 → 是高斯平滑的锅, 不是 Born 的锅.
  提炼3 - real FNO 对照: 必须有等配 real FNO + softmax 作为锚点. 绝对阈值 4.5
          是空中楼阁; GO/NO-GO 用相对判决 (L_A <= L_B + 0.3).

架构:
  线 A (探针): 复 FNO (全复 FFT + 复谱滤波器 R(k)) + modReLU + Born next-byte NLL
  线 B (对照): real FNO (rfft 同 exp34) + GELU + softmax CE, 参数量匹配

  输入: byte seq (B, L) → 高斯核连续嵌入 (sigma 控制扩散) → 网格场 (B, L, d)
  演化: 单层 [spectral_conv + local_conv + 非线性 + skip]
  输出: 读 position L-1 (最后一个字节位置) 的隐状态 → 预测 next byte

关键物理:
  - Born 概率: P(j) = |<psi_pred, C_j>|^2 / sum_m |<psi_pred, C_m>|^2
  - 坍缩指纹: psi_pred → 0 时 Born 退化为均匀分布, loss 卡 ln(256)=5.545
  - modReLU: g(z) = tanh(|z|) * z/|z|, 幅度硬钳 <=1, 相位保留. 结构性消灭爆炸,
              坍缩留作 NO-GO 信号.
  - 复数自动微分: PyTorch 原生 cfloat; NLL 是实数 (|.|^2 实数), Wirtinger 自动.

复用 exp34 数据加载 (v28 parquet → UTF-8 bytes → tensor).
不碰 manifesto, 不碰 exp02/exp03, 不加任何正则/闭包.
"""
from __future__ import annotations

import argparse
import math
import time
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_train.parquet"
VAL_PARQUET = PROJECT_ROOT / "crystalllm" / "data" / "processed" / "v28_val.parquet"
VOCAB_SIZE = 256
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIFORM_LOSS = math.log(VOCAB_SIZE)  # ln(256) = 5.5451774


# ============================================================================
# 数据加载 (复用 exp34 模式: v28 parquet → UTF-8 bytes → tensor[int64])
# ============================================================================
def load_data(train_n_bytes=2_000_000, val_n_bytes=100_000):
    print(f"[load] reading {TRAIN_PARQUET.name} ...")
    train_text = "\n".join(pd.read_parquet(TRAIN_PARQUET)["text"].astype(str).tolist())
    val_text = "\n".join(pd.read_parquet(VAL_PARQUET)["text"].astype(str).tolist())
    train_bytes = train_text.encode("utf-8")[:train_n_bytes]
    val_bytes = val_text.encode("utf-8")[:val_n_bytes]
    print(f"[load] train bytes: {len(train_bytes):,}  val bytes: {len(val_bytes):,}")
    return (torch.tensor(list(train_bytes), dtype=torch.int64),
            torch.tensor(list(val_bytes), dtype=torch.int64))


def get_batch(ids, bs, sl, device=DEVICE):
    """随机采样: x = ids[s:s+sl], y = ids[s+1:s+sl+1] (next-byte targets)."""
    n = len(ids) - sl - 1
    starts = torch.randint(0, n, (bs,))
    x = torch.stack([ids[s:s + sl] for s in starts])
    y = torch.stack([ids[s + 1:s + sl + 1] for s in starts])
    return x.to(device), y.to(device)


# ============================================================================
# 高斯核连续嵌入: byte → 复数嵌入 → 高斯扩散到网格 → 连续波场初始条件
# ============================================================================
class GaussianComplexEmbed(nn.Module):
    """高斯核连续复数嵌入.

    每个 byte b 在位置 i 的复数嵌入 e_b ∈ C^d, 经高斯核扩散到相邻网格点:
        Ψ_0(x_m) += e_b * exp(-(x_m - x_i)^2 / (2 σ^2))
    σ=0 → delta 编码 (每字节只写自己的网格点, 近离散, 对照组).
    σ>0 → 字节扩散到相邻网格, 形成连续波场.

    表示: (B, L, d, 2) 复数, [.,.,.,0]=real, [.,.,.,1]=imag.

    输出同时提供 real 视图 (B, L, 2d) 给 real FNO 对照线复用同一嵌入源.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d=32, sigma=0.0, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.sigma = float(sigma)
        self.max_len = max_len
        # 复数嵌入码本: (vocab, d, 2). 实部=语义强度, 虚部=初始相位.
        self.emb = nn.Parameter(torch.randn(vocab_size, d, 2) * 0.05)

        # 预计算高斯核权重矩阵 (L, L): spread[i, m] = exp(-(m-i)^2 / (2σ^2)) / Z
        # L 维度上归一化使每行和为 1 (能量不因扩散而膨胀).
        self.register_buffer("spread", self._build_spread(max_len, sigma), persistent=False)

    @staticmethod
    def _build_spread(L, sigma):
        if sigma <= 0:
            return torch.eye(L)  # delta: 不扩散
        idx = torch.arange(L).float()
        # (L_i, L_m) 距离矩阵
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L)
        w = torch.exp(-(diff ** 2) / (2.0 * sigma * sigma))
        w = w / w.sum(dim=1, keepdim=True)  # 每行归一化 (能量守恒扩散)
        return w

    def forward(self, byte_ids):
        """
        Args:
            byte_ids: (B, L) long
        Returns:
            field_complex: (B, L, d, 2) 复数场 (高斯扩散后)
            field_real: (B, L, 2d) 实数视图 = [real, imag] 拼接 (给 real FNO 用)
        """
        B, L = byte_ids.shape
        e = self.emb[byte_ids]  # (B, L, d, 2)
        # 高斯扩散: 在 L 维度上做 spread @ e. spread: (L_i, L_m), e: (B, L_i, d, 2)
        # field[b, m, d, p] = sum_i spread[i, m] * e[b, i, d, p]
        field = torch.einsum('im,bldp->bmdp', self.spread.to(e.device), e)  # (B, L, d, 2)
        field_real = torch.cat([field[..., 0], field[..., 1]], dim=-1)  # (B, L, 2d)
        return field, field_real


# ============================================================================
# 线 A: 复数 FNO + modReLU + Born next-byte NLL
# ============================================================================
class ComplexSpectralConv1d(nn.Module):
    """复数 1D 谱卷积 (FNO §3.2, 复值权重).

    x ∈ C^{L × C} → RFFT(x) ∈ C^{L//2+1 × C} → 截断前 modes 个频率
    → 复数线性映射 (复 einsum) → IRFFT → C^{L × C_out}

    复数权重 W: (C_out, C_in, modes) ∈ C. PyTorch cfloat.
    """

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(out_ch, in_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        """x: (B, C, L) complex → (B, C_out, L) complex."""
        B, C, L = x.shape
        x_ft = torch.fft.fft(x, dim=-1)  # full complex FFT (not rfft — x is complex)
        # FFT 频率索引: 0..L//2 为正频, L//2+1..L-1 为负频 (共轭对称已破坏).
        # FNO 截断: 保留前 modes 个低频正频率.
        eff_modes = min(self.modes, L // 2)
        x_ft_low = x_ft[:, :, :eff_modes]  # (B, C, eff_modes)
        # 复数 einsum: out[o, m] = sum_i W[o,i,m] * x_ft[i, m]
        out_ft = torch.einsum('oim,bim->bom', self.weights[:, :, :eff_modes].to(x_ft.dtype), x_ft_low)
        # 重建完整频谱: 低频填充 out_ft, 其余置零
        full_ft = torch.zeros(B, out_ft.size(1), L, dtype=x_ft.dtype, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return torch.fft.ifft(full_ft, dim=-1)  # (B, C_out, L) complex


def complex_modrelu(z, b=0.0):
    """modReLU: g(z) = tanh(|z|) * z / |z|.

    幅度硬钳 <= 1 (tanh 上界), 相位完全保留. 结构性消灭爆炸.
    b 是可学习偏置 (默认 0, 探针不加). 用 tanh 而非 relu 保证可导 (|z|=0 处).
    """
    mag = torch.abs(z)
    # 安全相位提取: z / max(|z|, eps), 避免 |z|=0 处 NaN
    phase = z / torch.clamp(mag, min=1e-8)
    return torch.tanh(mag) * phase


class ComplexFNOBlock(nn.Module):
    """单层复 FNO 块: spectral_conv + local_conv(复) + modReLU + skip.

    输入/输出: (B, C, L) complex.
    """

    def __init__(self, channels, modes, L):
        super().__init__()
        self.spec = ComplexSpectralConv1d(channels, channels, modes)
        # 局部线性 (复数权重): 作用在通道维, 1x1 等价复线性层.
        # 复数 Conv1d = local_re(z.real) + i·local_im(z.imag).
        self.local_re = nn.Conv1d(channels, channels, 1)
        self.local_im = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x: (B, C, L) complex
        h_spec = self.spec(x)
        h_local = self.local_re(x.real) + 1j * self.local_im(x.imag)
        h = h_spec + h_local
        # modReLU (复数非线性, 保相位, 幅度 tanh 钳 <= 1)
        h = complex_modrelu(h)
        # 复数 LayerNorm: 对实部虚部分别在 L 维归一化 (保相位, 稳定谱能量数值).
        re = F.layer_norm((x + h).real, (x + h).real.shape[-1:])
        im = F.layer_norm((x + h).imag, (x + h).imag.shape[-1:])
        return re + 1j * im


class ComplexFNOProbe(nn.Module):
    """线 A: 复 FNO + Born next-byte.

    forward(byte_ids) → 复数 logits (B, L, V) via Born 内积.
    实际只用最后一个位置预测 next-byte, 但全程计算便于诊断.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, sigma=0.0,
                 n_layers=1, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.embed = GaussianComplexEmbed(vocab_size, d, sigma, max_len)
        # 频域需要 pos 编码? FNO 对绝对位置不敏感. 加 learned pos (复数).
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d, 2) * 0.02)
        self.blocks = nn.ModuleList([
            ComplexFNOBlock(channels=d, modes=modes, L=max_len) for _ in range(n_layers)
        ])
        # Born 码本: (V, d, 2). 与输入嵌入独立 (测量基底 ≠ 输入基底, 更公平).
        self.codebook = nn.Parameter(torch.randn(vocab_size, d, 2) * 0.05)

    def forward(self, byte_ids):
        """byte_ids: (B, L) → logits_via_born: (B, L, V) real (log-prob 归一化前的得分)."""
        field, _ = self.embed(byte_ids)  # (B, L, d, 2) 复数场
        field = field + self.pos_embed  # broadcast (1, L, d, 2)
        # 转 (B, d, L) cfloat
        z = torch.view_as_complex(field.contiguous())  # (B, L, d) cfloat
        z = z.permute(0, 2, 1)  # (B, d, L) cfloat
        for blk in self.blocks:
            z = blk(z)  # (B, d, L) cfloat
        z = z.permute(0, 2, 1)  # (B, L, d) cfloat
        # Born 内积: psi_pred (B, L, d) ⊗ codebook (V, d) → |<.,.>|^2 → softmax
        C = torch.view_as_complex(self.codebook.contiguous())  # (V, d) cfloat
        # inner[b, l, v] = <psi[b,l], C[v]> = sum_d conj(psi) * C  (Hermitian)
        # 注意 Born 法则: P(v) ∝ |<Phi_v, psi>|^2. 这里 Phi_v = C[v].
        inner = torch.einsum('bld,vd->blv', torch.conj(z), C)  # (B, L, V) cfloat
        born_scores = (inner.real ** 2 + inner.imag ** 2)  # |<.,.>|^2
        # 返回 born_scores (非负实数). CE 用 log(softmax(born_scores)).
        # 直接传给 F.cross_entropy 会当 logits; 但 born_scores>=0, 非 logit.
        # 正确: NLL = -log( P(target) ) = -log( born_scores[target] / sum born_scores )
        return born_scores  # (B, L, V)


# ============================================================================
# 线 B: real FNO + GELU + softmax CE (对照, 复用 exp34 SpectralConv1d 风格)
# ============================================================================
class RealSpectralConv1d(nn.Module):
    """real 1D 谱卷积 (exp34 同款 rfft)."""

    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * modes)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, L) real
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1)
        n_freq = x_ft.size(-1)
        eff_modes = min(self.modes, n_freq)
        x_ft_trunc = x_ft[:, :, :eff_modes]
        out_ft = torch.einsum('bct,cot->bot', x_ft_trunc, self.weights[:, :, :eff_modes])
        full_ft = torch.zeros(B, out_ft.size(1), n_freq, dtype=torch.cfloat, device=x.device)
        full_ft[:, :, :eff_modes] = out_ft
        return torch.fft.irfft(full_ft, n=L, dim=-1)


class RealFNOProbe(nn.Module):
    """线 B: real FNO + GELU + softmax CE. 参数量匹配线 A."""

    def __init__(self, vocab_size=VOCAB_SIZE, d=32, modes=16, sigma=0.0,
                 n_layers=1, max_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        # real 模型谱卷积宽度 = d (匹配复 FNO 的谱卷积秩). 两者都用 cfloat 谱权重,
        # 所以 spectral-conv rank (通道数) 才是容量真正所在, 必须对齐为 d 而非 2d.
        # embed 仍用复数 (B,L,d,2), 取 real 视图 (B,L,2d) 后投影到 d.
        self.d_real = d
        self.embed = GaussianComplexEmbed(vocab_size, d, sigma, max_len)
        self.embed_proj = nn.Linear(2 * d, d)  # 把 (B,L,2d) 投影到 (B,L,d) 匹配谱卷积宽度
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, self.d_real) * 0.02)
        self.specs = nn.ModuleList([RealSpectralConv1d(self.d_real, modes) for _ in range(n_layers)])
        self.locals = nn.ModuleList([nn.Conv1d(self.d_real, self.d_real, 1) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.d_real) for _ in range(n_layers)])
        self.head = nn.Linear(self.d_real, vocab_size)

    def forward(self, byte_ids):
        _, field_real = self.embed(byte_ids)  # (B, L, 2d)
        h = self.embed_proj(field_real) + self.pos_embed  # (B, L, d)
        h = h.permute(0, 2, 1)  # (B, d, L)
        for spec, local, norm in zip(self.specs, self.locals, self.norms):
            h1 = spec(h) + local(h)
            h = norm(h1.transpose(1, 2)).transpose(1, 2) + h
            h = F.gelu(h)
        logits = self.head(h.transpose(1, 2))  # (B, L, V)
        return logits


# ============================================================================
# Loss: Born NLL (线 A) 与标准 CE (线 B)
# ============================================================================
def born_nll_loss(born_scores, targets):
    """Born 法则 NLL: -log( born_scores[target] / sum_v born_scores[v] ).

    born_scores: (B, L, V) 非负实数 (|<psi, C_v>|^2).
    targets: (B, L) long.
    """
    probs = born_scores / (born_scores.sum(dim=-1, keepdim=True) + 1e-8)
    log_probs = torch.log(probs + 1e-8)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
    return nll.mean()


def ce_loss(logits, targets):
    """标准 cross-entropy (线 B)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


# ============================================================================
# 评估: TF next-byte PPL + 梯度范数 (诊断 NaN/消失)
# ============================================================================
@torch.no_grad()
def eval_val_loss(model, val_ids, byte_ids_to_state, loss_fn, seq_len=256, n_seqs=40):
    """Teacher-forced val loss (next-byte, 全序列平均)."""
    model.eval()
    n = len(val_ids) - seq_len - 1
    starts = torch.randint(0, n, (n_seqs,))
    total_loss, total_tokens = 0.0, 0
    for s in starts:
        x = val_ids[s:s + seq_len].unsqueeze(0).to(DEVICE)
        y = val_ids[s + 1:s + 1 + seq_len].unsqueeze(0).to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item() * seq_len
        total_tokens += seq_len
    model.train()
    return total_loss / total_tokens


def grad_norm(model):
    """全局梯度范数 (诊断 NaN / 消失).

    复数梯度 (.grad 为 cfloat) 用 abs() 取模长 (Wirtinger 梯度的正确范数).
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        sq = (g.real ** 2 + g.imag ** 2) if g.is_complex() else (g ** 2)
        total += sq.sum().item()
    return math.sqrt(total)


# ============================================================================
# Main
# ============================================================================
def run_one(mode, sigma, args):
    """跑单个 (mode, sigma) 组合."""
    tag = f"{mode}_sigma{str(sigma).replace('.', 'p')}"
    print(f"\n{'='*70}\n[{tag}] mode={mode}  sigma={sigma}\n{'='*70}")

    torch.manual_seed(42)
    Model = ComplexFNOProbe if mode == "complex" else RealFNOProbe
    loss_fn = born_nll_loss if mode == "complex" else ce_loss

    model = Model(
        vocab_size=VOCAB_SIZE, d=args.d_model, modes=args.modes,
        sigma=sigma, n_layers=args.n_layers, max_len=args.seq_len,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {mode}  params: {n_params:,} ({n_params/1e6:.3f}M)  "
          f"d={args.d_model} modes={args.modes} layers={args.n_layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.peak_lr,
                            weight_decay=0.01, betas=(0.9, 0.95))

    train_ids, val_ids = args._train_ids, args._val_ids

    results = {
        "mode": mode, "sigma": sigma, "params": n_params,
        "uniform_loss": round(UNIFORM_LOSS, 4),
        "d_model": args.d_model, "modes": args.modes, "n_layers": args.n_layers,
        "seq_len": args.seq_len, "peak_lr": args.peak_lr,
        "trace": [], "nan_step": None, "diverged": False,
    }
    t0 = time.time()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(1, args.steps + 1):
        lr = args.peak_lr * min(step / args.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_ids, args.batch_size, args.seq_len)
        out = model(x)
        loss = loss_fn(out, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! NaN/Inf at step {step}, loss={loss.item()}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break

        opt.zero_grad()
        loss.backward()
        gnorm = grad_norm(model)
        if math.isnan(gnorm) or math.isinf(gnorm):
            print(f"  !!! grad NaN/Inf at step {step}, aborting")
            results["nan_step"] = step
            results["diverged"] = True
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            mem = (torch.cuda.max_memory_allocated() / 1e9) if DEVICE == "cuda" else 0
            print(f"  step {step:>4}/{args.steps}  lr={lr:.1e}  loss={loss.item():.4f}  "
                  f"|g|={gnorm:.3e}  mem={mem:.2f}GB  t={time.time()-t0:.0f}s", flush=True)

        if step in args.eval_steps:
            vl = eval_val_loss(model, val_ids, None, loss_fn, seq_len=args.seq_len)
            results["trace"].append({
                "step": step, "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "grad_norm": round(gnorm, 4),
            })
            print(f"  >>> val@{step}: {vl:.4f}  (uniform={UNIFORM_LOSS:.4f})", flush=True)

    results["total_time_s"] = round(time.time() - t0, 2)
    results["final_val_loss"] = results["trace"][-1]["val_loss"] if results["trace"] else None
    results["final_grad_norm"] = results["trace"][-1]["grad_norm"] if results["trace"] else None

    out_json = RESULTS_DIR / f"{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[saved] -> {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Exp04: Born probe (complex FNO vs real FNO)")
    parser.add_argument("--mode", choices=["complex", "real"], required=True)
    parser.add_argument("--sigma", type=float, default=0.0, choices=[0.0, 0.5, 1.0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--d_model", type=int, default=32, help="复数维度 d (real 线用 2d)")
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--peak_lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, nargs="+",
                        default=[200, 500, 1000, 2000, 3000])
    args = parser.parse_args()

    train_ids, val_ids = load_data()
    args._train_ids, args._val_ids = train_ids, val_ids
    run_one(args.mode, args.sigma, args)


if __name__ == "__main__":
    main()
