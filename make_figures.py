"""
Make presentation-ready figures (16:9 PNG) from the saved experiment data.

Run:  .venv\\Scripts\\python.exe make_figures.py
Out:  figures/*.png

Data sources (all produced by the training / eval scripts):
  results/summary.json                 - every model on the held-out sets
  results/pretrain*_metrics.json       - Step 1 curves
  results/grpo_*_metrics.json          - Step 2 curves
  data/eval_*.json                     - test-set composition
The exploration numbers (fig 5) come from the sampling probe run during the project
(32 samples x 210 b-words per temperature), copied below.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- palette (validated categorical order, light mode) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # blue, orange, aqua, violet
CRIT = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 11,
    "lines.linewidth": 2,
})
FIG = (12.8, 7.2)   # 16:9


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, subtitle=None):
    if subtitle:
        fig.text(0.012, 0.012, subtitle, color=MUTED, fontsize=10, ha="left", va="bottom")
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def suptitle(fig, title, sub=None):
    fig.text(0.04, 0.95, title, fontsize=22, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=13, color=INK2, va="top")


summary = {r["file"]: r for r in load("summary.json")}
final = summary["grpo_final.pt"]
pre = summary["pretrain.pt"]


# ---------------------------------------------------------------- fig 01: headline
def fig_headline():
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "MiniGPT: reverse a string, then skip the bomb \"b\"",
             "600,832-parameter decoder trained from scratch (Step 1) and fine-tuned with GRPO (Step 2)")
    tiles = [
        ("100%", "Step 1 reversal\nexact match", "1,350 held-out words", S1),
        ("0%", "\"b\" leak after GRPO", "was 100% before RL", S3),
        ("39.0%", "skip-b exact match\non words with \"b\"", "was 0% before RL", S2),
        ("96.5%", "no-b words still\nreversed correctly", "skill kept after RL", S4),
    ]
    for i, (big, label, note, col) in enumerate(tiles):
        x0 = 0.04 + i * 0.235
        ax = fig.add_axes([x0, 0.22, 0.215, 0.56]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="white", ec=GRID, lw=1.2, transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0.08, 0.86), 0.18, 0.025, color=col, transform=ax.transAxes))
        ax.text(0.08, 0.62, big, fontsize=46, fontweight="bold", color=INK, transform=ax.transAxes)
        ax.text(0.08, 0.30, label, fontsize=14, color=INK, transform=ax.transAxes, va="center")
        ax.text(0.08, 0.10, note, fontsize=11, color=INK2, transform=ax.transAxes)
    save(fig, "01_headline.png", "Source: results/summary.json (greedy decoding on held-out test sets)")


# ---------------------------------------------------------------- fig 02: architecture
def fig_architecture():
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "GPT model configuration", "Decoder-only Transformer, GPT-2 style (pre-LayerNorm), character-level")
    ax = fig.add_axes([0.03, 0.05, 0.52, 0.78]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(y, h, text, fc, ec=GRID, tc=INK, fs=12, bold=False):
        ax.add_patch(FancyBboxPatch((0.2, y), 9.6, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(5, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal")

    def arrow(y0, y1):
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))

    ax.text(5, 9.65, "<BOS> a b c d e <SEP>  (token ids)", ha="center", color=INK2, fontsize=12, family="DejaVu Sans Mono")
    arrow(9.4, 8.95)
    box(8.15, 0.8, "Token embedding (30 x 128)  +  Position embedding (15 x 128)", "white")
    arrow(8.15, 7.7)
    ax.add_patch(FancyBboxPatch((0.0, 3.2), 10.0, 4.45, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="#eef4fc", ec=S1, lw=1.4, ls="--"))
    ax.text(9.9, 7.35, "x 3 layers", ha="right", color=S1, fontsize=12, fontweight="bold")
    box(6.15, 0.8, "LayerNorm  →  Causal self-attention (4 heads x 32)", "white")
    ax.text(5, 5.75, "+ residual", ha="center", color=INK2, fontsize=10)
    box(4.4, 0.8, "LayerNorm  →  MLP 128 → 512 → 128 (GELU)", "white")
    ax.text(5, 4.0, "+ residual", ha="center", color=INK2, fontsize=10)
    arrow(3.2, 2.75)
    box(1.95, 0.8, "LayerNorm  →  Linear 128 → 30 (tied with token embedding)", "white")
    arrow(1.95, 1.5)
    ax.text(5, 1.05, "softmax  →  next-token probabilities", ha="center", color=INK2, fontsize=12)

    tx = fig.add_axes([0.58, 0.08, 0.39, 0.74]); tx.axis("off")
    rows = [("Vocabulary", "30 = 4 special + a–z"), ("Context length", "15 tokens"),
            ("Layers", "3"), ("Attention heads", "4"), ("Embedding dim", "128"),
            ("MLP hidden", "512 (4x), GELU"), ("Positional enc.", "learned, absolute"),
            ("Weight tying", "yes"), ("Dropout", "0"), ("Parameters", "600,832")]
    for i, (k, v) in enumerate(rows):
        y = 0.95 - i * 0.095
        tx.text(0.0, y, k, fontsize=14, color=INK2, transform=tx.transAxes, va="center")
        tx.text(1.0, y, v, fontsize=14, color=INK, transform=tx.transAxes, va="center", ha="right",
                fontweight="bold" if k == "Parameters" else "normal")
        tx.plot([0, 1], [y - 0.047, y - 0.047], color=GRID, lw=0.8, transform=tx.transAxes)
    save(fig, "02_model_config.png", "Source: common.py (GPTConfig, MiniGPT)")


# ---------------------------------------------------------------- fig 03: dataset
def fig_dataset():
    ev_r = json.load(open(os.path.join(ROOT, "data", "eval_reverse.json")))
    ev_b = json.load(open(os.path.join(ROOT, "data", "eval_bomb.json")))
    lens = [2, 3, 4, 5, 6]
    cr = [sum(len(w) == L for w in ev_r) for L in lens]
    cb = [sum(len(w) == L for w in ev_b) for L in lens]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Training and evaluation data",
             "Training words are drawn fresh every step; any word in a test set is rejected (no leakage)")
    ax = fig.add_axes([0.06, 0.13, 0.5, 0.65])
    w = 0.36
    xs = range(len(lens))
    b1 = ax.bar([x - w / 2 - 0.01 for x in xs], cr, w, color=S1, label=f"eval_reverse ({len(ev_r):,} words)")
    b2 = ax.bar([x + w / 2 + 0.01 for x in xs], cb, w, color=S2, label=f"eval_bomb ({len(ev_b):,} words, all contain \"b\")")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 5, f"{int(r.get_height())}",
                    ha="center", color=INK2, fontsize=11)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"length {L}" for L in lens])
    ax.set_ylabel("words in test set"); ax.set_ylim(0, 430); ax.set_yticks(range(0, 301, 50))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    ax.set_title("Held-out test sets by word length")
    tx = fig.add_axes([0.62, 0.13, 0.35, 0.65]); tx.axis("off")
    lines = [("Sequence format", True),
             ("<BOS> h e l l o <SEP> o l l e h <EOS>", False),
             ("", False),
             ("Training", True),
             ("3,000 steps x 256 random words", False),
             ("= 768,000 examples (Step 1)", False),
             ("", False),
             ("Loss masking", True),
             ("Loss only on tokens after <SEP>;", False),
             ("input letters are random, so", False),
             ("predicting them is wasted effort", False),
             ("", False),
             ("Why fewer length-2 words?", True),
             ("Only 26² = 676 exist", False)]
    for i, (t, head) in enumerate(lines):
        tx.text(0, 1 - i * 0.072, t, transform=tx.transAxes, fontsize=13 if head else 12,
                color=INK if head else INK2, fontweight="bold" if head else "normal",
                family="DejaVu Sans Mono" if t.startswith("<BOS>") else "DejaVu Sans")
    save(fig, "03_datasets.png", "Source: data/eval_reverse.json, data/eval_bomb.json, common.py")


# ---------------------------------------------------------------- fig 04: pretraining
def fig_pretrain():
    ls = load("pretrain_metrics.json"); nols = load("pretrain_nols_metrics.json")
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Step 1: pretraining reaches 100% by step 250",
             "Exact match on 1,350 held-out words; 100% at every length 2–6 after 3,000 steps (~104 s on CPU)")
    ax1 = fig.add_axes([0.06, 0.12, 0.42, 0.64])
    ax1.plot(nols["log"]["step"], nols["log"]["loss"], color=S1, label="no label smoothing")
    ax1.plot(ls["log"]["step"], ls["log"]["loss"], color=S2, label="label smoothing 0.1")
    ax1.set_yscale("log"); ax1.set_xlabel("training step"); ax1.set_ylabel("loss (answer tokens, log scale)")
    ax1.set_title("Training loss")
    ax1.text(250, ls["log"]["loss"][-1] * 0.3, "stays at 0.64 (smoothing floor)", color=INK, ha="left", fontsize=11)
    ax1.text(150, 3e-9, "loss → ~0: over-confident", color=INK, ha="left", fontsize=11)
    ax1.legend(loc="lower right")
    ax2 = fig.add_axes([0.56, 0.12, 0.40, 0.64])
    ax2.plot(ls["log"]["eval_step"], [v * 100 for v in ls["log"]["eval_exact"]], color=S2,
             marker="o", ms=7, mec=SURFACE, mew=2)
    ax2.set_ylim(-3, 105); ax2.set_xlabel("training step"); ax2.set_ylabel("held-out exact match (%)")
    ax2.set_title("Held-out accuracy (label smoothing 0.1)")
    ax2.annotate("100% at step 250", xy=(250, 100), xytext=(800, 70), color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    save(fig, "04_pretraining.png", "Source: results/pretrain_metrics.json, results/pretrain_nols_metrics.json")


# ---------------------------------------------------------------- fig 05: reward design
def fig_reward():
    cases = [("edcba", "writes the b", -0.03), ("edcxa", "replaces b with x", 0.48),
             ("edc", "stops at the b", 0.63), ("edcaa", "skips b, too long", 0.68),
             ("edca", "correct", 2.00)]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Reward design: every step toward the answer scores higher",
             "Input  a b c d e   →   target  e d c a")
    ax = fig.add_axes([0.25, 0.30, 0.42, 0.50])
    ys = list(range(len(cases)))[::-1]
    cols = [CRIT if r < 0 else (S3 if r >= 2 else S1) for _, _, r in cases]
    ax.barh(ys, [c[2] for c in cases], height=0.6, color=cols)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([f"{o:<6}  {d}" for o, d, _ in cases], family="DejaVu Sans Mono", fontsize=12)
    for y, (_, _, r) in zip(ys, cases):
        ax.text(r + (0.05 if r >= 0 else -0.05), y, f"{r:+.2f}", va="center",
                ha="left" if r >= 0 else "right", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.4, 2.4); ax.grid(axis="y", visible=False); ax.set_xlabel("reward")
    ax.set_title("reward_shaped() for five possible outputs")
    tx = fig.add_axes([0.71, 0.30, 0.27, 0.50]); tx.axis("off")
    terms = [("Formula", True), ("+ correct prefix / length", False), ("  (0 to 1)", False),
             ("− 0.5 per \"b\" written", False), ("− 0.5 x |length error| / length", False),
             ("+ 1.0 if exactly right", False), ("= −1 if malformed output", False)]
    for i, (t, h) in enumerate(terms):
        tx.text(0, 1 - i * 0.12, t, transform=tx.transAxes, fontsize=14 if h else 13,
                color=INK if h else INK2, fontweight="bold" if h else "normal")
    bx = fig.add_axes([0.04, 0.05, 0.92, 0.17]); bx.axis("off")
    bx.text(0, 0.85, "Earlier reward versions were exploited (reward hacking):", fontsize=13, fontweight="bold",
            color=INK, transform=bx.transAxes)
    hacks = ["\"+1 if no b\"  →  outputs anything without b; reversal collapses to 30%",
             "similarity ratio  →  writes x instead of b:  mb → xm",
             "prefix only (no length term)  →  stops at the b:  mb → (empty)"]
    for i, h in enumerate(hacks):
        bx.text(0.02, 0.55 - i * 0.27, "• " + h, fontsize=12, color=INK2, transform=bx.transAxes)
    save(fig, "05_reward_design.png", "Source: train_grpo.py (reward_shaped, reward_similarity, reward_naive)")


# ---------------------------------------------------------------- fig 06: exploration
def fig_exploration():
    temps = ["T = 1.0", "T = 1.5", "T = 2.0", "T = 3.0"]
    nols = [0.000, 0.000, 0.000, 0.000]
    ls = [0.074, 0.060, 0.015, 0.060]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "RL can only reinforce what the model sometimes does",
             "How often sampling produces the correct skip-b answer (32 samples per word, 210 words containing b)")
    ax = fig.add_axes([0.07, 0.14, 0.55, 0.62])
    xs = range(len(temps)); w = 0.36
    ax.bar([x - w / 2 - 0.01 for x in xs], nols, w, color=S1, label="pretrained without label smoothing")
    ax.bar([x + w / 2 + 0.01 for x in xs], ls, w, color=S2, label="pretrained with label smoothing 0.1")
    for x, v in zip(xs, nols):
        ax.text(x - w / 2 - 0.01, v + 0.002, "0", ha="center", color=INK2, fontsize=11)
    for x, v in zip(xs, ls):
        ax.text(x + w / 2 + 0.01, v + 0.002, f"{v:.3f}%", ha="center", color=INK, fontsize=11)
    ax.set_xticks(list(xs)); ax.set_xticklabels(temps); ax.set_ylim(0, 0.09)
    ax.set_ylabel("samples with the exact skip-b answer (%)"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right"); ax.set_title("Correct answers found by sampling")
    tx = fig.add_axes([0.67, 0.14, 0.31, 0.62]); tx.axis("off")
    pts = [("GRPO result from each start", True),
           ("", False),
           ("No label smoothing", True),
           ("All 16 samples identical → advantage 0", False),
           ("skip-b after 400 steps: 0.0%", False),
           ("", False),
           ("Label smoothing 0.1", True),
           ("Rare correct samples get reinforced", False),
           (f"skip-b after 4,000 steps: {final['b_words_skip_exact']*100:.1f}%", False),
           ("", False),
           ("Greedy accuracy is 100% for both,", False),
           ("so the difference only shows in RL.", False)]
    for i, (t, h) in enumerate(pts):
        tx.text(0, 1 - i * 0.08, t, transform=tx.transAxes, fontsize=13,
                color=INK if h else INK2, fontweight="bold" if h else "normal")
    save(fig, "06_exploration.png", "Source: sampling probe on checkpoints/pretrain*.pt; results/summary.json")


# ---------------------------------------------------------------- fig 07: GRPO curves
def fig_grpo_curves():
    a = load("grpo_long_lr5e-5_metrics.json")["log"]; b = load("grpo_final_metrics.json")["log"]
    steps = a["eval_step"] + [s + 2000 for s in b["eval_step"][1:]]
    bomb = a["eval_bomb_exact"] + b["eval_bomb_exact"][1:]
    nob = a["eval_nob_exact"] + b["eval_nob_exact"][1:]
    leak = a["eval_b_leak"] + b["eval_b_leak"][1:]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Step 2: GRPO removes every \"b\" quickly, then plateaus near 40%",
             "Held-out evaluation every 25 steps, greedy decoding; final model = 4,000 steps")
    ax = fig.add_axes([0.07, 0.12, 0.74, 0.64])
    ax.plot(steps, [v * 100 for v in nob], color=S1, label="no-b words: reversal exact")
    ax.plot(steps, [v * 100 for v in bomb], color=S2, label="b-words: skip-b exact")
    ax.plot(steps, [v * 100 for v in leak], color=S3, label="b-leak rate")
    ax.axvline(2000, color=MUTED, lw=1, ls=":")
    ax.text(2020, 60, "continued training\n(same KL anchor)", color=INK2, fontsize=11)
    ax.set_ylim(-3, 105); ax.set_xlim(0, 4000)
    ax.set_xlabel("GRPO step"); ax.set_ylabel("%")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    for ys, col, lab in [(nob, S1, f"{nob[-1]*100:.1f}%"), (bomb, S2, f"{bomb[-1]*100:.1f}%"),
                         (leak, S3, f"{leak[-1]*100:.1f}%")]:
        ax.plot(steps[-1], ys[-1] * 100, "o", color=col, ms=8, mec=SURFACE, mew=2)
        ax.text(steps[-1] + 60, ys[-1] * 100, lab, color=INK, fontsize=12, va="center", clip_on=False)
    ax.set_title("Held-out accuracy during RL")
    save(fig, "07_grpo_training.png", "Source: results/grpo_long_lr5e-5_metrics.json + results/grpo_final_metrics.json")


# ---------------------------------------------------------------- fig 08: model comparison
def fig_comparison():
    order = [("pretrain.pt", "Pretrained (start)"),
             ("grpo_naive.pt", "Naive reward"),
             ("grpo_similarity.pt", "Similarity reward"),
             ("grpo_nols.pt", "Shaped, no-LS start"),
             ("grpo_shaped.pt", "Shaped, 600 steps"),
             ("grpo_ent05.pt", "Shaped + entropy 0.05"),
             ("grpo_long_lr5e-5.pt", "Shaped, 2,000 steps"),
             ("grpo_final.pt", "Shaped, 4,000 (final)")]
    rows = [(lab, summary[f]) for f, lab in order if f in summary]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Fine-tuned models compared on the same test sets",
             "Goal: high skip-b accuracy while keeping no-b reversal near 100%")
    ax = fig.add_axes([0.24, 0.12, 0.68, 0.66])
    ys = list(range(len(rows)))[::-1]; h = 0.36
    nob = [r["no_b_words_exact"] * 100 for _, r in rows]
    skb = [r["b_words_skip_exact"] * 100 for _, r in rows]
    ax.barh([y + h / 2 + 0.01 for y in ys], nob, h, color=S1, label="no-b words: reversal exact")
    ax.barh([y - h / 2 - 0.01 for y in ys], skb, h, color=S2, label="b-words: skip-b exact")
    for y, v in zip(ys, nob):
        ax.text(v + 1, y + h / 2 + 0.01, f"{v:.1f}", va="center", fontsize=10, color=INK2)
    for y, v in zip(ys, skb):
        ax.text(v + 1, y - h / 2 - 0.01, f"{v:.1f}", va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([lab for lab, _ in rows])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0, 110); ax.set_xlabel("exact match (%)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    save(fig, "08_model_comparison.png", "Source: results/summary.json")


# ---------------------------------------------------------------- fig 09: per length
def fig_per_length():
    lens = ["2", "3", "4", "5", "6"]
    series = [("grpo_similarity.pt", "Similarity reward", S3),
              ("grpo_shaped.pt", "Shaped, 600 steps", S1),
              ("grpo_final.pt", "Shaped, 4,000 steps (final)", S2)]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Longer words are harder: each skipped b shifts every later letter",
             "Skip-b exact match on words containing b, by word length")
    ax = fig.add_axes([0.07, 0.12, 0.66, 0.64])
    nudge = {"grpo_similarity.pt": -3.5, "grpo_shaped.pt": 2.5, "grpo_final.pt": 1.5}
    for f, lab, col in series:
        v = [summary[f]["b_words_per_length"][L] * 100 for L in lens]
        ax.plot(lens, v, color=col, marker="o", ms=8, mec=SURFACE, mew=2, label=lab)
        ax.text(4.12, v[-1] + nudge[f], lab, color=INK, fontsize=11, va="center")
    ax.set_ylim(0, 105); ax.set_xlabel("word length"); ax.set_ylabel("skip-b exact match (%)")
    ax.set_xlim(-0.2, 4.1); ax.legend(loc="upper right")
    ax.set_title("Accuracy by length")
    v = summary["grpo_final.pt"]["b_words_per_length"]
    ax.annotate(f"{v['2']*100:.1f}%", xy=(0, v["2"] * 100), xytext=(0.25, 92), color=INK, fontsize=12)
    tx = fig.add_axes([0.80, 0.12, 0.18, 0.64]); tx.axis("off")
    tx.text(0, 0.98, "Typical errors", fontsize=13, fontweight="bold", color=INK, transform=tx.transAxes)
    errs = [("gob", "og", "oo"), ("hqbs", "sqh", "sqq"), ("jbqis", "siqj", "siqi"), ("osbg", "gso", "gss")]
    tx.text(0, 0.90, "input   target  output", fontsize=11, color=INK2, family="DejaVu Sans Mono", transform=tx.transAxes)
    for i, (a, b, c) in enumerate(errs):
        tx.text(0, 0.82 - i * 0.07, f"{a:<7} {b:<7} {c}", fontsize=11, color=INK, family="DejaVu Sans Mono",
                transform=tx.transAxes)
    tx.text(0, 0.45, "The model drops the b\nbut does not shift its\ncopy position, so the\nnext letter repeats.",
            fontsize=11, color=INK2, transform=tx.transAxes, va="top")
    save(fig, "09_per_length.png", "Source: results/summary.json; error examples from checkpoints/grpo_shaped.pt")


# ---------------------------------------------------------------- fig 10: entropy collapse
def fig_entropy():
    a = load("grpo_long_lr5e-5_metrics.json")["log"]; e = load("grpo_ent05_metrics.json")["log"]

    def smooth(v, k=8):
        return [sum(v[max(0, i - k + 1): i + 1]) / len(v[max(0, i - k + 1): i + 1]) for i in range(len(v))]
    fig = plt.figure(figsize=FIG)
    suptitle(fig, "Entropy collapse: the groups stop disagreeing",
             "GRPO learns only from differences inside a group of 16 answers; average within-group reward std")
    ax = fig.add_axes([0.07, 0.12, 0.66, 0.64])
    ax.plot(a["step"], smooth(a["reward_std"]), color=S1, label="no entropy bonus")
    ax.plot(e["step"], smooth(e["reward_std"]), color=S2, label="entropy bonus 0.05")
    ax.set_xlabel("GRPO step"); ax.set_ylabel("within-group reward std (moving avg of 8 logs)")
    ax.set_ylim(0, 0.8); ax.set_xlim(0, 2000); ax.legend(loc="upper right")
    ax.set_title("Diversity of sampled answers")
    tx = fig.add_axes([0.78, 0.12, 0.20, 0.64]); tx.axis("off")
    lines = [("Result on b-words", True), ("", False),
             ("No bonus, 2,000 steps", False), (f"{summary['grpo_long_lr5e-5.pt']['b_words_skip_exact']*100:.1f}%", True),
             ("", False), ("Entropy 0.05, 1,000 steps", False),
             (f"{summary['grpo_ent05.pt']['b_words_skip_exact']*100:.1f}%", True), ("", False),
             ("More randomness kept", False), ("groups diverse, but random", False),
             ("noise rarely finds the", False), ("systematic shift.", False)]
    for i, (t, h) in enumerate(lines):
        tx.text(0, 1 - i * 0.075, t, transform=tx.transAxes, fontsize=16 if (h and "%" in t) else 12,
                color=INK if h else INK2, fontweight="bold" if h else "normal")
    save(fig, "10_entropy_collapse.png", "Source: results/grpo_long_lr5e-5_metrics.json, results/grpo_ent05_metrics.json")


if __name__ == "__main__":
    fig_headline(); fig_architecture(); fig_dataset(); fig_pretrain(); fig_reward()
    fig_exploration(); fig_grpo_curves(); fig_comparison(); fig_per_length(); fig_entropy()
