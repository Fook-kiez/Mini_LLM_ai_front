"""
common.py — shared code: vocabulary, dataset, MiniGPT model, generation, evaluation.

Task (Step 1):  <BOS> h e l l o <SEP> o l l e h <EOS>
Task (Step 2):  <BOS> a b c d e <SEP> e d c a <EOS>   (skip the letter "b")
"""
import json
import math
import os
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1) Vocabulary / tokenizer  (one token = one character)
# ----------------------------------------------------------------------------
CHARS = "abcdefghijklmnopqrstuvwxyz"
SPECIAL = ["<PAD>", "<BOS>", "<SEP>", "<EOS>"]
ITOS = SPECIAL + list(CHARS)                 # id -> token
STOI = {s: i for i, s in enumerate(ITOS)}    # token -> id
PAD, BOS, SEP, EOS = 0, 1, 2, 3
VOCAB_SIZE = len(ITOS)                       # 30

MIN_LEN, MAX_LEN = 2, 6
# longest sequence: BOS + 6 chars + SEP + 6 chars + EOS = 15 tokens
BLOCK_SIZE = 1 + MAX_LEN + 1 + MAX_LEN + 1
MAX_NEW_TOKENS = MAX_LEN + 1                 # answer (<=6 chars) + EOS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
for _d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return " ".join(ITOS[i] for i in ids)


def target_reverse(x):
    """Step 1 target: reversed string."""
    return x[::-1]


def target_skip_b(x):
    """Step 2 target: reversed string with every 'b' removed."""
    return x[::-1].replace("b", "")


def prompt_ids(x):
    return [BOS] + encode(x) + [SEP]


def full_ids(x, y):
    return [BOS] + encode(x) + [SEP] + encode(y) + [EOS]


# ----------------------------------------------------------------------------
# 2) Dataset
# ----------------------------------------------------------------------------
def random_word(rng, lo=MIN_LEN, hi=MAX_LEN, must_have_b=False):
    L = rng.randint(lo, hi)
    w = [rng.choice(CHARS) for _ in range(L)]
    if must_have_b and "b" not in w:
        # put 1..2 'b' at random positions
        for _ in range(rng.randint(1, min(2, L))):
            w[rng.randrange(L)] = "b"
    return "".join(w)


def build_eval_sets(seed=1234, n_per_len=300, n_b_per_len=200):
    """Fixed, held-out evaluation sets (saved to data/ so they are reproducible).

    eval_reverse.json : random strings, length 2..6 (n_per_len each)
    eval_bomb.json    : strings that contain at least one 'b' (n_b_per_len each)
    Every string in these sets is NEVER used for training.
    """
    path_r = os.path.join(DATA_DIR, "eval_reverse.json")
    path_b = os.path.join(DATA_DIR, "eval_bomb.json")
    if os.path.exists(path_r) and os.path.exists(path_b):
        return json.load(open(path_r)), json.load(open(path_b))
    rng = random.Random(seed)
    ev_r, ev_b, seen = [], [], set()
    for L in range(MIN_LEN, MAX_LEN + 1):
        # length-2 strings only have 26^2 = 676 possibilities, so take fewer
        n_r = min(n_per_len, 150) if L == 2 else n_per_len
        n_b = min(n_b_per_len, 40) if L == 2 else n_b_per_len
        cnt = 0
        while cnt < n_r:
            w = random_word(rng, L, L)
            if w not in seen:
                seen.add(w); ev_r.append(w); cnt += 1
        cnt = 0
        while cnt < n_b:
            w = random_word(rng, L, L, must_have_b=True)
            if w not in seen:
                seen.add(w); ev_b.append(w); cnt += 1
    json.dump(ev_r, open(path_r, "w"))
    json.dump(ev_b, open(path_b, "w"))
    return ev_r, ev_b


class WordSampler:
    """Draws fresh random training words on the fly, excluding held-out eval words."""

    def __init__(self, held_out, seed=0):
        self.held_out = set(held_out)
        self.rng = random.Random(seed)

    def sample(self, must_have_b=False):
        while True:
            w = random_word(self.rng, must_have_b=must_have_b)
            if w not in self.held_out:
                return w


def make_sft_batch(words, target_fn=target_reverse, device="cpu"):
    """Build (input, target) tensors for next-token prediction.

    We only compute the loss on the ANSWER part (tokens after <SEP>, incl. <EOS>),
    because the input letters are random and cannot be predicted anyway.
    Ignored positions get label -100.
    """
    T = BLOCK_SIZE - 1
    X = torch.full((len(words), T), PAD, dtype=torch.long)
    Y = torch.full((len(words), T), -100, dtype=torch.long)
    for i, w in enumerate(words):
        ids = full_ids(w, target_fn(w))
        inp, tgt = ids[:-1], ids[1:]
        sep_pos = len(w) + 1                    # index of <SEP> in ids
        X[i, : len(inp)] = torch.tensor(inp)
        lab = torch.tensor(tgt)
        lab[:sep_pos] = -100                    # don't learn to predict the random input
        Y[i, : len(tgt)] = lab
    return X.to(device), Y.to(device)


# ----------------------------------------------------------------------------
# 3) MiniGPT  (decoder-only Transformer, GPT-2 style, pre-LayerNorm)
# ----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # causal mask: token t can only look at tokens 0..t
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying (like GPT-2)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def save_model(model, path, extra=None):
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    m = MiniGPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["state"])
    return m


# ----------------------------------------------------------------------------
# 4) Generation (batched, prompts may have different lengths)
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate(model, words, greedy=True, temperature=1.0, device="cpu", prefixes=None, max_new=None):
    """Returns a LongTensor [B, BLOCK_SIZE] (prompt + completion, right-padded with PAD)
    and a list with the prompt length of each row.

    Sequences are right-padded. Because attention is causal, padding to the RIGHT
    never influences earlier positions, so we can read the logits at each row's own
    last position.

    prefixes (optional): list of strings already written after <SEP> (counted as part of
    the prompt, so plen includes them). Used by GRPO "start-state resets".
    """
    model.eval()
    B = len(words)
    seq = torch.full((B, BLOCK_SIZE), PAD, dtype=torch.long, device=device)
    plen = []
    for i, w in enumerate(words):
        p = prompt_ids(w) + (encode(prefixes[i]) if prefixes else [])
        seq[i, : len(p)] = torch.tensor(p, device=device)
        plen.append(len(p))
    cur = torch.tensor(plen, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    rows = torch.arange(B, device=device)
    for _ in range(max_new or MAX_NEW_TOKENS):
        logits, _ = model(seq)
        last = logits[rows, cur - 1]                    # [B, V]
        if greedy:
            nxt = last.argmax(-1)
        else:
            probs = F.softmax(last / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        can_write = (~done) & (cur < BLOCK_SIZE)
        seq[rows[can_write], cur[can_write]] = nxt[can_write]
        done = done | (nxt == EOS) | (cur + 1 >= BLOCK_SIZE)
        cur = cur + can_write.long()
        if done.all():
            break
    return seq, plen


def parse_completion(seq_row, plen):
    """Convert generated token ids (after <SEP>) into a string.
    Returns None if the output is malformed (no <EOS>, or a special token in the middle)."""
    out = []
    for t in seq_row[plen:].tolist():
        if t == EOS:
            return "".join(out)
        if t < len(SPECIAL):          # PAD/BOS/SEP inside the answer => invalid
            return None
        out.append(ITOS[t])
    return None                       # never produced <EOS>


@torch.no_grad()
def predict(model, words, device="cpu", batch_size=512):
    preds = []
    for i in range(0, len(words), batch_size):
        chunk = words[i: i + batch_size]
        seq, plen = generate(model, chunk, greedy=True, device=device)
        preds += [parse_completion(seq[j], plen[j]) for j in range(len(chunk))]
    return preds


def evaluate(model, words, target_fn, device="cpu"):
    """Exact-match accuracy (overall + per length), char accuracy, b-leak rate."""
    preds = predict(model, words, device)
    per_len = {}
    exact = char_ok = char_tot = invalid = has_b = 0
    examples = []
    for w, p in zip(words, preds):
        t = target_fn(w)
        ok = (p == t)
        exact += ok
        L = len(w)
        a = per_len.setdefault(L, [0, 0]); a[0] += ok; a[1] += 1
        if p is None:
            invalid += 1
            char_tot += len(t)
        else:
            has_b += ("b" in p)
            char_tot += max(len(t), len(p))
            char_ok += sum(1 for c1, c2 in zip(p, t) if c1 == c2)
        if len(examples) < 12:
            examples.append({"input": w, "target": t, "pred": p, "correct": ok})
    n = len(words)
    return {
        "n": n,
        "exact_match": exact / n,
        "char_accuracy": char_ok / max(char_tot, 1),
        "invalid_rate": invalid / n,
        "b_leak_rate": has_b / n,
        "per_length": {str(k): v[0] / v[1] for k, v in sorted(per_len.items())},
        "examples": examples,
    }


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
