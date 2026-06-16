"""
CrystaLLM 最小原型 — 验证扩散定位 → AR 寻路的信息相变
  阶段 I：5 步把高熵噪声凝缩为低熵语义锚点 z
  阶段 II：z 作为首 token embedding 注入 GRU，后续字符走 teacher-forced AR
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

ANCHORS = ['cat', 'dog', 'fish']
W2I = {'<pad>': 0, '<eos>': 1}
for w in ANCHORS:
    for c in w: W2I.setdefault(c, len(W2I))
I2W = {i: c for c, i in W2I.items()}; V = len(W2I); D_Z, D_H = 12, 96

class Diffusion(nn.Module):                                              # 阶段 I：扩散模块
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D_Z*2, D_H), nn.SiLU(),
                              nn.Linear(D_H, D_H), nn.SiLU(), nn.Linear(D_H, D_Z))
    def step(s, z, t): return z - 0.3 * s.net(torch.cat([z, t.expand(D_Z)]))
    def sample(s, K=5):
        z = torch.randn(D_Z)
        for i in range(K-1, -1, -1): z = s.step(z, torch.tensor(i/K))
        return z

class Decoder(nn.Module):                                                # 阶段 II：z-prefix AR
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D_H); s.z2h = nn.Linear(D_Z, D_H)
        s.rnn = nn.GRU(D_H, D_H, batch_first=True); s.head = nn.Linear(D_H, V)
    def gen(s, z, n=6):
        x = s.z2h(z).view(1, 1, -1); o, h = s.rnn(x); out = []
        for _ in range(n):
            tok = s.head(o[:, -1]).argmax(-1).clamp(max=V-1)
            out.append(tok.item())
            if tok.item() == W2I['<eos>']: break
            x = s.emb(tok).view(1, 1, -1); o, h = s.rnn(x, h)
        return out

# 联合训练：解码器学会 z → 类别词；扩散学会 z_noisy → z_tgt
diff, dec = Diffusion(), Decoder()
opt = torch.optim.Adam(list(diff.parameters()) + list(dec.parameters()), 1e-3)
for ep in range(4000):
    k = torch.randint(0, 3, (1,)).item()
    z_tgt = (torch.eye(3)[k].repeat(1, D_Z//3+1)[:, :D_Z].squeeze()) * 4
    z_in = z_tgt + 0.1 * torch.randn(D_Z)
    target = [W2I[c] for c in ANCHORS[k]] + [W2I['<eos>']]
    z_emb = dec.z2h(z_in).view(1, 1, -1)
    tok_embs = dec.emb(torch.tensor([target[:-1]]))
    x = torch.cat([z_emb, tok_embs], dim=1)                              # (1, 1+len-1, D_H)
    o, _ = dec.rnn(x)
    z_noisy = z_tgt + 1.5 * torch.randn(D_Z)
    loss = F.cross_entropy(dec.head(o).reshape(-1, V), torch.tensor(target)) \
         + 0.1 * (diff.step(z_noisy, torch.tensor(0.0)) - z_tgt).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 演示信息相变
print("CrystaLLM min-proto - phase-transition demo")
for k, name in enumerate(ANCHORS):
    z_anchor = (torch.eye(3)[k].repeat(1, D_Z//3+1)[:, :D_Z].squeeze()) * 4
    z_noisy = z_anchor + 2.0 * torch.randn(D_Z)
    z_clean = diff.step(z_noisy, torch.tensor(0.0))
    word = ''.join(I2W[t] for t in dec.gen(z_clean) if I2W[t] != '<eos>')
    print(f"  anchor={name:4s} | ||z_noise||={z_noisy.norm():.2f} -> ||z_clean||={z_clean.norm():.2f} | gen='{word}'")
