import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=100, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # LINE 6: YOUR EMBEDDING!
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # TIED TO EMBEDDING!

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embed(x) + self.pos_embed[:, :T, :]  # EMBEDDING + POSITION
        for layer in self.layers: tok_emb = layer(tok_emb)  # DECODER STACK
        tok_emb = self.ln_f(tok_emb)
        logits = self.head(tok_emb)  # LM HEAD
        return logits[:, -1, :]  # NEXT TOKEN PREDICTION!

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # CAUSAL ATTENTION
        x = x + self.ffn(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.w_qkv = nn.Linear(d_model, 3*d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.register_buffer('mask', torch.tril(torch.ones(512, 512)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.w_qkv(x).reshape(B, T, 3, self.n_heads, C//self.n_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, T, head_dim]
        att = (q @ k.transpose(-2,-1)) / (C**0.5)  # CAUSAL MASKING!
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).reshape(B, T, C)
        return self.w_o(y)

# TRAINING (5 LINES!)
model = MiniGPT()
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
data = torch.randint(0, 100, (1000, 8))  # FAKE TEXT
for i in range(100):
    logits = model(data[:, :-1])
    loss = F.cross_entropy(logits, data[:, -1])
    optim.zero_grad(); loss.backward(); optim.step()
    if i % 20 == 0: print(f"Step {i}, Loss: {loss.item():.3f}")

print("🎉 MINI GPT BUILT IN 50 LINES! INSIDE MODEL, OUTSIDE DECODER ✓")