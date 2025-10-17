import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request  # DOWNLOAD SHAKESPEARE!

# DOWNLOAD TINY SHAKESPEARE (1 LINE!)
urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "shakespeare.txt")

# LOAD & PREPARE DATA (5 LINES!)
with open('shakespeare.txt', 'r') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
train_data = data[:1000000].unsqueeze(0)  # 1M chars!
print(f"Vocab: {vocab_size} chars, Training on {len(train_data[0])} chars")

# MINI GPT CLASS (SAME AS BEFORE!)
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # INSIDE MODEL, OUTSIDE DECODER!
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.embed(x) + self.pos_embed[:, :T, :]
        for layer in self.layers: tok_emb = layer(tok_emb)
        tok_emb = self.ln_f(tok_emb)
        logits = self.head(tok_emb)
        return logits[:, -1, :]

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
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
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2,-1)) / (C**0.5)
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1,2).reshape(B, T, C)
        return self.w_o(y)

# TRAIN ON SHAKESPEARE! (GPU/CPU AUTO!)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniGPT(vocab_size).to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"\n� TRAINING ON {device.upper()}...")
block_size = 32  # Context length
for i in range(500):  # 500 steps
    # Sample batch
    ix = torch.randint(0, len(train_data[0]) - block_size, (64,))  # 64 sequences
    x = train_data[0, ix].unsqueeze(0)  # [1, 64, 32]
    y = train_data[0, ix+1].unsqueeze(0)
    
    logits = model(x.to(device))
    loss = F.cross_entropy(logits, y.to(device)[:, -1])
    optim.zero_grad(); loss.backward(); optim.step()
    
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.3f}")

# GENERATE SHAKESPEARE! (5 LINES!)
print("\n� YOUR GPT WRITES SHAKESPEARE...")
def generate(prompt="To be", length=100):
    x = torch.tensor([[stoi[c] for c in prompt]]).to(device)
    for _ in range(length):
        logit = model(x)
        next_token = torch.argmax(logit).unsqueeze(0)
        x = torch.cat([x, next_token.unsqueeze(1)], dim=1)
    return "".join([itos[i.item()] for i in x[0]])

# RUN QUERIES!
print("1:", generate("To be"))
print("2:", generate("Romeo:"))
print("3:", generate("King:"))
print("\n� MINI GPT + SHAKESPEARE COMPLETE!")