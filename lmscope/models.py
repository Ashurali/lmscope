"""Model architectures, copied from the HW3 code so its checkpoints load unchanged.

Parameter names must stay identical to HW3 (`embed`, `lstm`, `head` for the LSTM; `tok_embed`,
`pos_embed`, `blocks.N.{ln1,attn.qkv,attn.proj,ln2,ff.net}`, `ln_f`, `head` for the Transformer).
Two additions that add no parameters:
  * MultiHeadSelfAttention applies its softmax through an `AttentionSoftmax` submodule, so a forward
    hook on that submodule sees the attention weights.
  * CustomLSTMCell re-runs one nn.LSTM layer step by step from its weights and exposes the gates.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """Embedding -> multi-layer nn.LSTM -> dropout -> Linear."""

    def __init__(self, vocab_size, d_model=512, hidden_size=1024, n_layers=3, dropout=0.3, **_):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.head(self.drop(out)), hidden


class CustomLSTMCell:
    """One layer of an nn.LSTM, evaluated step by step with its gates exposed.

    nn.LSTM stacks the gate weights in the order (input, forget, cell candidate, output):
        gates = W_ih x_t + b_ih + W_hh h_{t-1} + b_hh
        i, f, o = sigmoid(...), g = tanh(...)
        c_t = f * c_{t-1} + i * g,   h_t = o * tanh(c_t)
    """

    def __init__(self, lstm, layer):
        self.w_ih = getattr(lstm, f"weight_ih_l{layer}")
        self.w_hh = getattr(lstm, f"weight_hh_l{layer}")
        self.b = getattr(lstm, f"bias_ih_l{layer}") + getattr(lstm, f"bias_hh_l{layer}")
        self.hidden_size = lstm.hidden_size

    @torch.no_grad()
    def run(self, x):
        """x: (T, input_size). Returns h (T, H), c (T, H) and {"input","forget","output","cell"} each (T, H)."""
        T, H = x.shape[0], self.hidden_size
        h = x.new_zeros(H)
        c = x.new_zeros(H)
        x_proj = x @ self.w_ih.T + self.b  # input part of all steps at once
        hs, cs, gi, gf, go, gg = [], [], [], [], [], []
        for t in range(T):
            gates = x_proj[t] + self.w_hh @ h
            i, f, g, o = gates.split(H)
            i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
            hs.append(h); cs.append(c); gi.append(i); gf.append(f); go.append(o); gg.append(g)
        stack = torch.stack
        return stack(hs), stack(cs), {"input": stack(gi), "forget": stack(gf), "output": stack(go), "cell": stack(gg)}


class AttentionSoftmax(nn.Module):
    """Softmax over the key axis. A module of its own so forward hooks can read attention weights."""

    def forward(self, scores):
        return F.softmax(scores, dim=-1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.softmax = AttentionSoftmax()
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = self.softmax(scores)  # (B, heads, T, T)
        out = self.attn_drop(attn) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, context_len, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class CausalTransformer(nn.Module):
    """Decoder-only Transformer with learned positional embeddings."""

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, context_len=256,
                 dropout=0.1, **_):
        super().__init__()
        self.context_len = context_len
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, context_len, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.shape[1]
        h = self.drop(self.tok_embed(x) + self.pos_embed(torch.arange(T, device=x.device)))
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def build_model(config):
    """Builds an untrained model from a checkpoint's model_config."""
    kind = config["model_type"]
    if kind == "lstm":
        return LSTMModel(**config)
    if kind == "transformer":
        return CausalTransformer(**config)
    raise ValueError(f"unknown model_type {kind!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
