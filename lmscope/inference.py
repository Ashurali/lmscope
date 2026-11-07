"""Step-by-step inference with access to model internals.

`Inspector` keeps one token sequence (the prompt plus generated tokens). "Position p" means the
prediction of token p from the tokens before it, so p runs from 1 to len(sequence); p = len(sequence)
is the next, not-yet-sampled token. The model always sees at most `context_len` tokens before p.
"""
import math

import torch
import torch.nn.functional as F

from .models import CausalTransformer, CustomLSTMCell, LSTMModel, build_model, count_params
from .tokenizers import load_tokenizer

LN2 = math.log(2)


def load_checkpoint(path, model_type=None, tokenizer_path=None, tokenizer_type=None, device=None):
    """Builds model and tokenizer from a checkpoint in the lmscope format (see README)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in ckpt or "model_config" not in ckpt:
        raise ValueError(f"{path} is not an lmscope checkpoint: it needs 'model_state_dict' and 'model_config'")
    config = dict(ckpt["model_config"])
    if model_type and model_type != config["model_type"]:
        raise ValueError(f"--model-type {model_type} but the checkpoint holds a {config['model_type']}")
    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tokenizer = load_tokenizer(tokenizer_type, tokenizer_path, ckpt)
    if tokenizer.vocab_size != config["vocab_size"]:
        raise ValueError(f"tokenizer has {tokenizer.vocab_size} tokens, model expects {config['vocab_size']}")
    return model, tokenizer, config


def _round(t, digits=4):
    return [round(float(x), digits) for x in t]


class Inspector:
    def __init__(self, model, tokenizer, config, seed=0):
        self.model = model
        self.tok = tokenizer
        self.config = config
        self.context_len = config["context_len"]
        self.device = next(model.parameters()).device
        self.seed = seed
        self.prompt_ids = []
        self.ids = []
        self.records = {}  # position -> how the generated token there was sampled
        self._gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ sequence state
    def model_info(self):
        c = self.config
        return {"model_type": c["model_type"], "params": count_params(self.model), "n_layers": c["n_layers"],
                "n_heads": c.get("n_heads"), "hidden_size": c.get("hidden_size"), "d_model": c["d_model"],
                "vocab_size": c["vocab_size"], "context_len": self.context_len,
                "tokenizer_type": self.tok.kind, "device": str(self.device)}

    def set_prompt(self, text):
        ids = self.tok.encode(text)
        if not ids:
            raise ValueError("the prompt is empty")
        self.prompt_ids = ids
        return self.reset()

    def reset(self):
        self.ids = list(self.prompt_ids)
        self.records = {}
        self._gen = torch.Generator().manual_seed(self.seed)
        return self.state()

    def state(self):
        tokens = []
        for p, i in enumerate(self.ids):
            rec = self.records.get(p)
            tokens.append({"id": i, "token": self.tok.token_str(i), "generated": p >= len(self.prompt_ids),
                           "prob": rec["prob"] if rec else None, "loss": rec["loss"] if rec else None})
        losses = [r["loss"] for r in self.records.values()]
        return {"tokens": tokens, "prompt_len": len(self.prompt_ids), "length": len(self.ids),
                "generated": len(losses), "text": self.tok.decode(self.ids),
                "perplexity": math.exp(sum(losses) / len(losses)) if losses else None}

    def _window(self, position):
        start = max(0, position - self.context_len)
        return start, self.ids[start:position]

    # ------------------------------------------------------------------ forward passes
    @torch.no_grad()
    def _logits(self, ids):
        x = torch.tensor([ids], device=self.device)
        out = self.model(x)
        return (out[0] if isinstance(self.model, LSTMModel) else out)[0].float()

    def _distribution(self, logits, temperature, top_k, chosen=None):
        probs = F.softmax(logits / temperature, dim=-1).cpu()
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum()) / LN2
        top_p, top_i = probs.topk(min(top_k, probs.numel()))
        mass = torch.cumsum(probs.sort(descending=True).values, 0)
        out = {"temperature": temperature,
               "top": [{"id": int(i), "token": self.tok.token_str(i), "prob": round(float(p), 6)}
                       for p, i in zip(top_p, top_i)],
               "entropy_bits": round(entropy, 4), "max_entropy_bits": round(math.log2(probs.numel()), 4),
               "topk_mass": {k: round(float(mass[k - 1]), 4) for k in (1, 5, 10, 20) if k <= probs.numel()},
               "chosen": None}
        if chosen is not None:
            out["chosen"] = {"id": chosen, "token": self.tok.token_str(chosen),
                             "prob": round(float(probs[chosen]), 6),
                             "rank": int((probs > probs[chosen]).sum()) + 1}
        return out, probs

    @torch.no_grad()
    def _transformer_internals(self, ids, layer, head):
        """One forward pass with hooks: attention of every layer, residual-stream norms per layer."""
        model = self.model
        attn, norms, hooks = [], [], []
        for block in model.blocks:
            hooks.append(block.attn.softmax.register_forward_hook(lambda m, i, o: attn.append(o[0])))
            hooks.append(block.register_forward_hook(lambda m, i, o: norms.append(o[0].norm(dim=-1))))
        try:
            x = torch.tensor([ids], device=self.device)
            emb = model.tok_embed(x) + model.pos_embed(torch.arange(len(ids), device=self.device))
            norms.append(emb[0].norm(dim=-1))
            model(x)
        finally:
            for h in hooks:
                h.remove()
        # norms: embedding first, then the output of each block (hooks fire in block order).
        a = attn[layer]  # (heads, T, T)
        matrix = a.mean(0) if head == "mean" else a[int(head)]
        # Where each head of this layer looks, for the last query: mean attention distance in tokens.
        T = len(ids)
        dist = (torch.arange(T, device=a.device).flip(0).float() * a[:, -1, :]).sum(-1)
        return {"kind": "attention", "layer": layer, "head": head, "n_layers": len(attn), "n_heads": a.shape[0],
                "matrix": [_round(row, 4) for row in matrix.cpu()],
                "last_row_by_head": [_round(row, 4) for row in a[:, -1, :].cpu()],
                "head_mean_distance": _round(dist.cpu(), 2),
                "norm_labels": ["embedding"] + [f"block {i + 1}" for i in range(len(attn))],
                "norms": [_round(n.cpu(), 3) for n in norms]}

    @torch.no_grad()
    def _lstm_internals(self, ids, layer):
        """Re-runs the LSTM layer by layer with CustomLSTMCell to expose gates and states."""
        model = self.model
        x = model.embed(torch.tensor(ids, device=self.device))
        layers = []
        for l in range(model.lstm.num_layers):
            h, c, gates = CustomLSTMCell(model.lstm, l).run(x)
            layers.append((h, c, gates))
            x = h  # inter-layer dropout is off in eval mode
        h, c, gates = layers[layer]
        return {"kind": "gates", "layer": layer, "n_layers": len(layers), "hidden_size": h.shape[1],
                "gates": {name: _round(g[-1].cpu(), 3) for name, g in gates.items()},
                "gate_means": {name: round(float(g[-1].mean()), 4) for name, g in gates.items()},
                "gate_means_over_time": {name: _round(g.mean(-1).cpu(), 4) for name, g in gates.items()},
                "norm_labels": [f"h, layer {i + 1}" for i in range(len(layers))],
                "norms": [_round(hh.norm(dim=-1).cpu(), 3) for hh, _, _ in layers],
                "cell_norms": [_round(cc.norm(dim=-1).cpu(), 3) for _, cc, _ in layers]}

    # ------------------------------------------------------------------ public API
    def analyze(self, position=None, temperature=1.0, layer=0, head="mean", top_k=20):
        """Distribution and internals for predicting the token at `position` (default: the next one)."""
        position = len(self.ids) if position is None else int(position)
        if not 1 <= position <= len(self.ids):
            raise ValueError(f"position must be between 1 and {len(self.ids)}")
        start, window = self._window(position)
        chosen = self.ids[position] if position < len(self.ids) else None
        dist, _ = self._distribution(self._logits(window)[-1], temperature, top_k, chosen)
        if isinstance(self.model, CausalTransformer):
            internals = self._transformer_internals(window, int(layer), head)
        else:
            internals = self._lstm_internals(window, int(layer))
        internals["labels"] = [self.tok.token_str(i) for i in window]
        return {"position": position, "window_start": start, "distribution": dist, "internals": internals,
                "chosen_generated": position in self.records, "record": self.records.get(position)}

    def sample(self, temperature=1.0):
        """Samples the next token, appends it and returns how it was chosen."""
        position = len(self.ids)
        _, window = self._window(position)
        logits = self._logits(window)[-1].cpu()
        probs = F.softmax(logits / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1, generator=self._gen))
        p1 = F.softmax(logits, dim=-1)[nxt]  # loss is always reported at T = 1
        self.records[position] = {"prob": round(float(probs[nxt]), 6), "temperature": temperature,
                                  "loss": round(float(-torch.log(p1)), 5)}
        self.ids.append(nxt)
        return {"position": position, "id": nxt, "token": self.tok.token_str(nxt), **self.records[position]}

    @torch.no_grad()
    def score(self, text):
        """Per-token cross-entropy (T = 1) of a reference continuation of the prompt, teacher-forced."""
        ref = self.tok.encode(text)
        if not ref:
            raise ValueError("the reference text is empty")
        seq = self.prompt_ids + ref
        n_prompt = len(self.prompt_ids)
        if len(seq) - 1 <= self.context_len:
            pred = self._logits(seq[:-1])[n_prompt - 1:]  # one pass covers every prediction
        else:  # longer than the context: one sliding window per predicted token
            pred = torch.stack([self._logits(seq[max(0, p - self.context_len):p])[-1]
                                for p in range(n_prompt, len(seq))])
        targets = torch.tensor(seq[n_prompt:], device=pred.device)
        losses = F.cross_entropy(pred, targets, reduction="none").tolist()
        running, total = [], 0.0
        for k, l in enumerate(losses, 1):
            total += l
            running.append(round(math.exp(total / k), 4))
        return {"tokens": [self.tok.token_str(i) for i in ref], "losses": _round(losses, 5),
                "running_perplexity": running, "mean_loss": round(total / len(losses), 5),
                "perplexity": running[-1],
                "bits_per_char": round(total / LN2 / len(self.tok.decode(ref)), 4)}
