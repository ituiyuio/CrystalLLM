# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CrystaLLM v2 — 可控生成的"三相变"原型
  数据：3 簇 × 5 词 = 15 词（animal/color/fruit）
  架构：3D 潜变量 z + 扩散 + GRU 解码器（沿用 v1）
  新增：簇内生成、簇间插值、‖z‖ 相变曲线 + 3D 潜空间散点
"""
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42); np.random.seed(42)

CLUSTERS = {
    'animal': ['cat', 'dog', 'fish', 'bird', 'mouse'],
    'color':  ['red', 'blue', 'green', 'black', 'white'],
    'fruit':  ['apple', 'grape', 'lemon', 'peach', 'mango'],
}
NAMES = list(CLUSTERS.keys()); N_C = len(NAMES); N_PER = 5
WORDS = [w for ws in CLUSTERS.values() for w in ws]
W2I = {'<pad>': 0, '<eos>': 1, ' ': 2}
for w in WORDS:
    for c in w: W2I.setdefault(c, len(W2I))
I2W = {i: c for c, i in W2I.items()}; V = len(W2I)
D_Z, D_H = 3, 128

class Diffusion(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, D_H), nn.SiLU(),
                              nn.Linear(D_H, D_H), nn.SiLU(), nn.Linear(D_H, D_Z))
    def step(s, z, t): return z - 0.3 * s.net(torch.cat([z, t.expand(D_Z)]))
    def denoise(s, z, K=5):
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K))
        return z

class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D_H); s.z2h = nn.Linear(D_Z, D_H)
        s.rnn = nn.GRU(D_H, D_H, batch_first=True); s.head = nn.Linear(D_H, V)
    def gen(s, z, n=8):
        x = s.z2h(z).view(1, 1, -1); o, h = s.rnn(x); out = []
        for _ in range(n):
            tok = s.head(o[:, -1]).argmax(-1).clamp(max=V-1)
            out.append(tok.item())
            if tok.item() == W2I['<eos>']: break
            x = s.emb(tok).view(1, 1, -1); o, h = s.rnn(x, h)
        return out

# 锚点：簇中心 + 簇内确定性小偏移
def anc(w):
    i = WORDS.index(w); c, j = i // N_PER, i % N_PER
    z = torch.zeros(D_Z); z[c] = 4.0
    z += 0.4 * torch.tensor([np.cos(j*1.3), np.sin(j*1.3), np.cos(j*0.7)])
    return z

# 训练
diff, dec = Diffusion(), Decoder()
opt = torch.optim.Adam(list(diff.parameters()) + list(dec.parameters()), 1e-3)
for ep in range(6000):
    w = WORDS[torch.randint(0, len(WORDS), (1,)).item()]
    z_tgt = anc(w)
    z_in = z_tgt + 0.1 * torch.randn(D_Z)
    target = [W2I[c] for c in w] + [W2I['<eos>']]
    z_emb = dec.z2h(z_in).view(1, 1, -1)
    tok_embs = dec.emb(torch.tensor([target[:-1]]))
    x = torch.cat([z_emb, tok_embs], dim=1)
    o, _ = dec.rnn(x)
    z_noisy = z_tgt + 1.5 * torch.randn(D_Z)
    loss = F.cross_entropy(dec.head(o).reshape(-1, V), torch.tensor(target)) \
         + 0.1 * (diff.step(z_noisy, torch.tensor(0.0)) - z_tgt).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 评估 1：簇内生成（取每簇第一个词的 anchor，注入噪声再 denoise）
print("=== 簇内生成（噪声→凝缩→解码）===")
for w in ['cat', 'red', 'apple']:
    z_t = anc(w); z_n = z_t + 2.0 * torch.randn(D_Z)
    z_c = diff.step(z_n, torch.tensor(0.0))
    out = dec.gen(z_c)
    word = ''.join(I2W[t] for t in out if I2W[t] != '<eos>')
    print(f"  target={w:5s} | ||z_n||={z_n.norm():.2f} -> ||z_c||={z_c.norm():.2f} | gen='{word}'")

# 评估 2：簇间插值（animal → color）
print("\n=== 簇间插值（cat <-> red，α 从 0 到 1）===")
z_a, z_b = anc('cat'), anc('red')
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z = (1-a)*z_a + a*z_b
    out = dec.gen(z)
    word = ''.join(I2W[t] for t in out if I2W[t] != '<eos>')
    print(f"  α={a:.2f} | z={np.round(z.numpy(),2)} | gen='{word}'")

# 可视化
fig = plt.figure(figsize=(12, 5))
cmap = {'animal': '#e74c3c', 'color': '#2ecc71', 'fruit': '#3498db'}

# 左：3D 潜空间 + 插值路径
ax1 = fig.add_subplot(121, projection='3d')
for w in WORDS:
    i = WORDS.index(w); c = i // N_PER
    name = NAMES[c]; z = anc(w).numpy()
    ax1.scatter(z[0], z[1], z[2], c=cmap[name], s=80, edgecolors='k', linewidth=0.5)
    ax1.text(z[0], z[1], z[2], w, fontsize=7)
# 插值路径
interp = np.array([(1-a)*z_a.numpy() + a*z_b.numpy() for a in np.linspace(0, 1, 8)])
ax1.plot(interp[:,0], interp[:,1], interp[:,2], 'k--o', alpha=0.6, markersize=4, label='cat→red path')
ax1.set_title('3D Latent Space (15 anchors + interp path)')
ax1.set_xlabel('z₀'); ax1.set_ylabel('z₁'); ax1.set_zlabel('z₂')
ax1.legend(loc='upper left', fontsize=8)

# 右：‖z‖ 相变曲线
ax2 = fig.add_subplot(122)
cluster_of = {w: n for n, ws in CLUSTERS.items() for w in ws}
for w in ['cat', 'red', 'apple']:
    z = anc(w) + 2.0 * torch.randn(D_Z)
    norms = [z.norm().item()]
    for i in range(5):
        z = diff.step(z, torch.tensor(i/5))
        norms.append(z.norm().item())
    name = cluster_of[w]
    ax2.plot(range(6), norms, 'o-', color=cmap[name], label=f'{w} (noise→denoise)')
ax2.set_xlabel('denoising step (0=noise, 5=clean)')
ax2.set_ylabel('||z||')
ax2.set_title('Phase Transition Curve')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crystalllm/phase_transition.png', dpi=100, bbox_inches='tight')
print("\nPlot saved: crystalllm/phase_transition.png")
