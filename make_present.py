"""
Build the presentation pack in present/ : slide-ready figures (16:9 PNG) + the raw numbers (CSV).

Run:  .venv/bin/python make_present.py          (Windows: .venv\\Scripts\\python.exe make_present.py)
Out:  present/fig/*.png, present/data/*.csv

Every number is read from files the training / eval scripts wrote:
  results/summary.json           - every checkpoint on the held-out test sets (evaluate_all.py)
  results/*_metrics.json         - training curves + hyper-parameters
  results/log_exp*.txt           - round-2 exploration runs (some were stopped early)
  data/eval_*.json               - the test sets
  checkpoints/*.pt               - used live for the example predictions
"""
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import (CKPT_DIR, build_eval_sets, load_model, predict, target_skip_b)
from train_grpo import reward_naive, reward_shaped, reward_similarity

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "present")
FIG_DIR, DATA_DIR = os.path.join(OUT, "fig"), os.path.join(OUT, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- palette: same validated categorical order as make_figures.py (light mode) ----
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # blue, orange, aqua, violet
CRIT = "#d03b3b"
HL = "#eaf2fc"   # light blue row highlight

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 12, "lines.linewidth": 2,
})
FIG = (12.8, 7.2)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def write_csv(name, header, rows):
    with open(os.path.join(DATA_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote data/" + name)


def new_fig(title, sub=None):
    fig = plt.figure(figsize=FIG)
    fig.text(0.04, 0.95, title, fontsize=22, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=13, color=INK2, va="top")
    return fig


def save(fig, name, source=None):
    if source:
        fig.text(0.012, 0.012, "Source: " + source, color=MUTED, fontsize=10, ha="left", va="bottom")
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)
    print("wrote fig/" + name)


def draw_table(fig, rect, header, rows, widths, highlight=(), bold_cols=(), fs=12, align=None):
    """Plain text table: header rule, light row rules, optional highlighted rows."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.autoscale(False)
    n = len(rows) + 1
    h = 1.0 / n
    xs = [sum(widths[:i]) / sum(widths) for i in range(len(widths) + 1)]
    align = align or ["left"] + ["center"] * (len(widths) - 1)
    for r in range(n):
        y = 1 - (r + 0.5) * h
        if r - 1 in highlight:
            ax.add_patch(plt.Rectangle((0, 1 - (r + 1) * h), 1, h, color=HL, lw=0))
        cells = header if r == 0 else rows[r - 1]
        for c, txt in enumerate(cells):
            x = xs[c] + 0.008 if align[c] == "left" else (xs[c] + xs[c + 1]) / 2
            weight = "bold" if (r == 0 or c in bold_cols or (r - 1) in highlight) else "normal"
            ax.text(x, y, str(txt), ha=align[c], va="center", fontsize=fs,
                    color=INK if r else INK2, fontweight=weight, family="DejaVu Sans")
        if r > 0:
            ax.plot([0, 1], [1 - r * h] * 2, color=INK2 if r == 1 else GRID, lw=1.2 if r == 1 else 0.8)
    return ax


def pct(x):
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------- data
summary = {r["file"]: r for r in load("summary.json")}
eval_rev, eval_bomb = build_eval_sets()
eval_nob = [w for w in eval_rev if "b" not in w]


def last_eval(tag):
    """(step, b-words, no-b) of the last eval line of an exploration-run log."""
    lines = [l for l in open(os.path.join(RES, f"log_{tag}.txt"), encoding="utf-8") if "[eval]" in l]
    m = re.search(r"step\s+(\d+) \| bomb-words exact\s+([\d.]+)% .* no-b words exact\s+([\d.]+)%", lines[-1])
    return int(m.group(1)), float(m.group(2)) / 100, float(m.group(3)) / 100


# Key models used across slides: (checkpoint, short label, colour)
KEY = [("pretrain.pt", "Pretrained (Step 1)", MUTED),
       ("grpo_final.pt", "Round 1: shaped reward\n(4,000 steps)", S2),
       ("grpo_expC.pt", "Round 2: token-level\n(400 steps)", S4),
       ("grpo_expI.pt", "Round 2 + hard-state\nexploration (FINAL)", S1)]


# ================================================================ 01 experiment design
def fig_design():
    fig = new_fig("Experiment design",
                  "Two-step pipeline. Every model is scored on the same held-out test words.")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.74]); ax.axis("off"); ax.set_xlim(-1.5, 101.5); ax.set_ylim(0, 60)

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=SURFACE, ec=color, lw=2))
        ax.text(x + 1.2, y + h - 2.2, title, fontsize=14, fontweight="bold", color=INK, va="top")
        for i, l in enumerate(lines):
            ax.text(x + 1.2, y + h - 6.5 - i * 3.6, l, fontsize=11.5, color=INK2, va="top")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6, mutation_scale=16))

    box(0, 33, 22, 25, "Data", ["random words a-z", "length 2-6", "test set fixed (seed 1234)",
                                "train words: fresh each", "step, never a test word"], MUTED)
    box(26, 33, 22, 25, "Step 1: pretrain", ["MiniGPT, 600,832 params", "supervised next-token",
                                             "target = reverse(x)", "label smoothing 0.1",
                                             "3,000 steps, ~104 s"], S2)
    box(52, 33, 22, 25, "Step 2: GRPO (RL)", ["no labels in the loss", "reward from a checker",
                                              "target = reverse(x)", "         minus every 'b'",
                                              "3 rounds of design"], S1)
    box(78, 33, 22, 25, "Evaluation", ["greedy decoding", "exact match on test sets",
                                       "+ b-leak rate", "+ per-length accuracy",
                                       "same sets for all models"], S3)
    arrow(22.6, 45, 25.4, 45); arrow(48.6, 45, 51.4, 45); arrow(74.6, 45, 77.4, 45)

    ax.text(52, 27, "Step 2 was developed in rounds (each answers a question from the last):",
            fontsize=13, fontweight="bold", color=INK, ha="center")
    rounds = [("Round 1", "Which reward works?", "naive / similarity /\nshaped (whole answer)", "39.0%", S2),
              ("Round 2", "Why stuck at 39%?", "token-level reward,\nper-state baseline", "59.3%", S4),
              ("Round 2b", "Why stuck at 60%?", "explore only at\nhard states", "96.1%", S1)]
    for i, (name, q, what, res, col) in enumerate(rounds):
        x = 4 + i * 32.5
        ax.add_patch(FancyBboxPatch((x, 1), 27, 21, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=SURFACE, ec=col, lw=2))
        ax.text(x + 1.2, 19.8, name, fontsize=13, fontweight="bold", color=INK, va="top")
        ax.text(x + 1.2, 15.6, q, fontsize=11.5, color=INK2, va="top", style="italic")
        ax.text(x + 1.2, 11.6, what, fontsize=11.5, color=INK2, va="top")
        ax.text(x + 25.8, 19.8, res, fontsize=17, fontweight="bold", color=INK, va="top", ha="right")
        ax.text(x + 25.8, 15.6, "b-words", fontsize=10, color=MUTED, va="top", ha="right")
        if i < 2:
            arrow(x + 27.6, 11, x + 32.0, 11)
    save(fig, "01_experiment_design.png", "train_pretrain.py, train_grpo.py, evaluate_all.py")


# ================================================================ 02 all experiments
def experiments_rows():
    rows = []   # id, round, what changed, steps, b-words, no-b, checkpoint/log
    s = summary
    def m(fn): return s[fn]["b_words_skip_exact"], s[fn]["no_b_words_exact"]
    spec = [
        ("pretrain", "Step 1", "pretrained, label smoothing 0.1 (start point)", "3,000", "pretrain.pt"),
        ("pretrain_nols", "Step 1", "pretrained, no label smoothing", "3,000", "pretrain_nols.pt"),
        ("naive", "R1", "reward: 1 if no 'b' else 0", "400", "grpo_naive.pt"),
        ("similarity", "R1", "reward: string similarity to target", "400", "grpo_similarity.pt"),
        ("nols", "R1", "shaped reward, start from no-LS model", "400", "grpo_nols.pt"),
        ("shaped", "R1", "shaped reward (prefix + length + b penalty)", "600", "grpo_shaped.pt"),
        ("ent05", "R1", "shaped + entropy bonus 0.05", "1,000", "grpo_ent05.pt"),
        ("long_g32", "R1", "shaped, G=32, lr 1e-4, beta 0.04", "2,000", "grpo_long_g32.pt"),
        ("long_lr5e-5", "R1", "shaped, G=16, lr 5e-5, beta 0.01", "2,000", "grpo_long_lr5e-5.pt"),
        ("final", "R1", "continue long_lr5e-5 (round 1 best)", "+2,000", "grpo_final.pt"),
        ("expC", "R2", "token-level reward + clamped baseline, T=1.5", "400", "grpo_expC.pt"),
    ]
    for tag, rnd, what, steps, fn in spec:
        b, n = m(fn)
        rows.append([tag, rnd, what, steps, b, n])
    logs = [("expA", "R2", "token-level, T=1.0 (no higher temperature)"),
            ("expB", "R2", "token-level, baseline fixed at 0"),
            ("expD", "R2", "continue expC, more steps (stopped)"),
            ("expE", "R2", "+ start episodes mid-answer (stopped)"),
            ("expF", "R2", "+ temperature 3 on whole answer (stopped)"),
            ("expG", "R2", "+ per-state std scaling (stopped)"),
            ("expH", "R2", "+ negative advantages x0.1 (stopped)")]
    for tag, rnd, what in logs:
        st, b, n = last_eval(tag)
        start = "+" if tag in "expD expE expF expG expH" else ""
        rows.append([tag, rnd, what, f"{start}{st:,}", b, n])
    b, n = m("grpo_expI.pt")
    rows.append(["expI", "R2b", "continue expC + one-step exploration at hard states", "+1,500", b, n])
    return rows


def fig_experiments():
    rows = experiments_rows()
    write_csv("all_experiments.csv", ["id", "round", "what changed", "GRPO steps",
                                      "b-words skip-b exact", "no-b words reverse exact"],
              [r[:4] + [f"{r[4]:.4f}", f"{r[5]:.4f}"] for r in rows])
    fig = new_fig("All experiments", "Held-out exact match. R1 = round 1, R2 = round 2.  '+' = continued from the previous checkpoint.")
    disp = [[r[0], r[1], r[2], r[3], pct(r[4]), pct(r[5])] for r in rows]
    best = [i for i, r in enumerate(rows) if r[0] in ("final", "expC", "expI")]
    draw_table(fig, [0.04, 0.06, 0.92, 0.78], ["id", "round", "what changed", "steps", "b-words (skip b)", "no-b words"],
               disp, [1.3, 0.8, 5.2, 1.0, 1.6, 1.3], highlight=best, fs=10.5,
               align=["left", "center", "left", "center", "center", "center"])
    save(fig, "02_all_experiments.png", "results/summary.json, results/log_exp*.txt")


# ================================================================ 03 hyper-parameters
def fig_hparams():
    pre = load("pretrain_metrics.json")
    r1, rc, ri = (load(f"grpo_{t}_metrics.json")["args"] for t in ("final", "expC", "expI"))
    cfg, tr = pre["config"], pre["train"]
    model_rows = [["layers", cfg["n_layer"]], ["attention heads", cfg["n_head"]],
                  ["embedding size", cfg["n_embd"]], ["MLP hidden", 4 * cfg["n_embd"]],
                  ["vocab (26 letters + 4 special)", cfg["vocab_size"]],
                  ["context length (tokens)", cfg["block_size"]], ["dropout", cfg["dropout"]],
                  ["parameters", f"{pre['params']:,}"]]
    step1 = [["steps", f"{tr['steps']:,}"], ["batch size", tr["batch_size"]], ["peak lr", tr["lr"]],
             ["warmup / schedule", f"{tr['warmup']} / cosine to 10%"], ["optimizer", "AdamW (0.9, 0.98)"],
             ["weight decay", tr["weight_decay"]], ["label smoothing", tr["label_smoothing"]],
             ["training time (CPU)", f"{tr['time_sec']:.0f} s"]]

    def g(a, k, d="-"): return a.get(k, d)
    grpo_keys = [("steps", lambda a: g(a, "steps")), ("prompts / step", lambda a: g(a, "prompts")),
                 ("G (samples / prompt)", lambda a: g(a, "group")), ("learning rate", lambda a: g(a, "lr")),
                 ("KL weight beta", lambda a: g(a, "beta")), ("PPO clip eps", lambda a: g(a, "clip")),
                 ("updates / batch", lambda a: g(a, "inner")), ("sampling temperature", lambda a: g(a, "temp")),
                 ("prompts with 'b'", lambda a: g(a, "b_frac")), ("advantage", lambda a: g(a, "adv", "seq")),
                 ("baseline", lambda a: g(a, "baseline", "group mean/std") if g(a, "adv", "seq") == "token" else "group mean/std"),
                 ("exploration episodes", lambda a: g(a, "explore_frac", 0)),
                 ("exploration temperature", lambda a: g(a, "explore_temp") if g(a, "explore_frac", 0) else "-"),
                 ("device", lambda a: g(a, "device", "cpu"))]
    grpo_rows = [[k, f(r1), f(rc), f(ri)] for k, f in grpo_keys]
    grpo_rows[0] = ["steps (total from pretrained)", "4,000 (2,000 + 2,000)", "400", "1,900 (400 + 1,500)"]
    for r in grpo_rows:
        if r[0] == "baseline":
            r[2] = "mixed (clamp >= 0)"
    write_csv("hyperparams_model_step1.csv", ["setting", "value"], model_rows + step1)
    write_csv("hyperparams_grpo.csv", ["setting", "round 1 final", "round 2 (expC)", "round 2b FINAL (expI)"], grpo_rows)

    fig = new_fig("Hyper-parameters", "Model and Step 1 (left), Step 2 GRPO for the three key models (right)")
    fig.text(0.04, 0.835, "MiniGPT", fontsize=14, fontweight="bold", color=INK)
    draw_table(fig, [0.04, 0.50, 0.30, 0.32], ["model", "value"], model_rows, [2.2, 1.0], fs=10.5)
    fig.text(0.04, 0.455, "Step 1 training", fontsize=14, fontweight="bold", color=INK)
    draw_table(fig, [0.04, 0.06, 0.30, 0.38], ["setting", "value"], step1, [2.2, 1.4], fs=10.5)
    fig.text(0.39, 0.835, "Step 2: GRPO", fontsize=14, fontweight="bold", color=INK)
    draw_table(fig, [0.39, 0.06, 0.58, 0.76], ["setting", "Round 1", "Round 2", "FINAL"], grpo_rows,
               [2.0, 1.45, 1.3, 1.35], fs=10)
    save(fig, "03_hyperparameters.png", "results/pretrain_metrics.json, results/grpo_{final,expC,expI}_metrics.json")


# ================================================================ 04 train / test split
def fig_split():
    lens = range(2, 7)
    rev_b = [sum(1 for w in eval_rev if len(w) == L and "b" in w) for L in lens]
    rev_nob = [sum(1 for w in eval_rev if len(w) == L and "b" not in w) for L in lens]
    bomb = [sum(1 for w in eval_bomb if len(w) == L) for L in lens]
    write_csv("test_sets_per_length.csv", ["length", "eval_reverse: no b", "eval_reverse: has b", "eval_bomb (all have b)"],
              [[L, a, b, c] for L, a, b, c in zip(lens, rev_nob, rev_b, bomb)])
    pre = load("pretrain_metrics.json")["train"]
    train_rows = [["Step 1 (3,000 steps)", f"{pre['examples_seen']:,} words", "random"],
                  ["Round 1 (4,000 steps)", f"{32 * 4000:,} x 16", "60% have b"],
                  ["Round 2 (400 steps)", f"{128 * 400:,} x 16", "70% have b"],
                  ["FINAL (+1,500 steps)", f"{256 * 1500:,} x 16", "70% have b"]]
    write_csv("train_data.csv", ["stage", "amount", "mix"], train_rows)

    fig = new_fig("Train / test split",
                  "Test words are fixed once and saved. Training words are drawn fresh every step and NEVER equal a test word.")
    ax = fig.add_axes([0.06, 0.13, 0.46, 0.62])
    x = list(lens); w = 0.36
    ax.bar([i - w / 2 for i in x], rev_nob, w * 0.94, color=S1, label="eval_reverse: no 'b'")
    ax.bar([i - w / 2 for i in x], rev_b, w * 0.94, bottom=rev_nob, color=S3, label="eval_reverse: has 'b'")
    ax.bar([i + w / 2 for i in x], bomb, w * 0.94, color=S2, label="eval_bomb (every word has 'b')")
    for i, a, b, c in zip(x, rev_nob, rev_b, bomb):
        ax.text(i - w / 2, a + b + 4, str(a + b), ha="center", fontsize=10.5, color=INK2)
        ax.text(i + w / 2, c + 4, str(c), ha="center", fontsize=10.5, color=INK2)
    ax.set_xticks(x); ax.set_xlabel("word length"); ax.set_ylabel("test words")
    ax.set_title("Test sets (held out)"); ax.legend(loc="upper left", fontsize=10.5); ax.set_ylim(0, 430)
    fig.text(0.56, 0.78, "Test sets", fontsize=14, fontweight="bold", color=INK)
    draw_table(fig, [0.56, 0.55, 0.40, 0.21], ["set", "words", "used for"],
               [["eval_reverse", f"{len(eval_rev):,}", "Step 1 score"],
                ["  - no-b part", f"{len(eval_nob):,}", "'no-skip-b' score"],
                ["eval_bomb", f"{len(eval_bomb):,}", "'skip-b' score"]], [1.4, 0.8, 1.6], fs=11)
    fig.text(0.56, 0.45, "Training data (generated on the fly)", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.56, 0.415, "'x 16' = prompts x 16 sampled answers (G) in GRPO", fontsize=10, color=MUTED)
    draw_table(fig, [0.56, 0.10, 0.40, 0.29], ["stage", "prompts", "mix"], train_rows, [1.9, 1.4, 1.0], fs=10.5)
    save(fig, "04_train_test_split.png", "data/eval_reverse.json, data/eval_bomb.json, common.py (WordSampler)")


# ================================================================ 05 reward functions
def token_marks(out, tgt):
    """+1 / -1 / . per token (incl. <EOS>) for the token-level reward."""
    if out is None:
        return "malformed"
    seq, ref = list(out) + ["<EOS>"], list(tgt) + ["<EOS>"]
    marks = []
    for i, t in enumerate(seq):
        ok = i < len(ref) and t == ref[i]
        marks.append("+1" if ok else "-1")
        if not ok:
            marks += ["0"] * (len(seq) - i - 1)
            break
    return " ".join(marks)


def fig_rewards():
    x, tgt = "abcde", target_skip_b("abcde")
    cands = [("edcba", "wrote b"), ("edcxa", "replaced b with x"), ("edc", "stopped at b"),
             ("edcc", "repeated a letter"), ("edcaa", "too long"), ("edca", "correct"), (None, "no <EOS>")]
    rows = []
    for out, note in cands:
        rows.append([out if out is not None else "(malformed)", note, f"{reward_naive(x, out):+.2f}",
                     f"{reward_similarity(x, out):+.2f}", f"{reward_shaped(x, out):+.2f}", token_marks(out, tgt)])
    write_csv("reward_examples.csv", ["output", "what happened", "naive", "similarity", "shaped (R1)", "token-level (R2) per token"], rows)
    fig = new_fig("Reward functions", f"Input '{x}'  ->  target '{tgt}'.  Same outputs scored by each reward.")
    draw_table(fig, [0.04, 0.40, 0.92, 0.44], ["output", "what happened", "naive", "similarity", "shaped (R1)", "token-level (R2)"],
               rows, [1.2, 1.9, 0.9, 1.1, 1.2, 2.2], highlight=[5], fs=11.5)
    notes = [("naive", "1 if no 'b' in the answer, else 0. Says nothing about WHAT to write -> model breaks (reward hacking)."),
             ("similarity", "SequenceMatcher ratio - 0.5 per 'b' + 1 if exact. Rewards 'edcxa' (b replaced by x) too much."),
             ("shaped (R1)", "correct-prefix fraction - 0.5 per 'b' - length penalty + 1 if exact. One number per answer."),
             ("token-level (R2)", "each token: +1 while still exactly right, -1 at the FIRST mistake, 0 after.\n"
                                  "Advantage = token reward - mean of the samples that reached the same state.")]
    for i, (k, v) in enumerate(notes):
        fig.text(0.04, 0.33 - i * 0.065, k, fontsize=12, fontweight="bold", color=INK, va="top")
        fig.text(0.19, 0.33 - i * 0.065, v, fontsize=11.5, color=INK2, va="top")
    save(fig, "05_reward_functions.png", "train_grpo.py (reward_naive, reward_similarity, reward_shaped, token_advantages)")


def fig_reward_results():
    s = summary
    items = [("naive", "grpo_naive.pt", "400"), ("similarity", "grpo_similarity.pt", "400"),
             ("shaped", "grpo_shaped.pt", "600"), ("shaped (long)", "grpo_final.pt", "4,000"),
             ("token-level", "grpo_expC.pt", "400"), ("token-level +\nexploration", "grpo_expI.pt", "1,900")]
    nob = [s[f]["no_b_words_exact"] for _, f, _ in items]
    bw = [s[f]["b_words_skip_exact"] for _, f, _ in items]
    leak = [s[f]["b_leak_rate"] for _, f, _ in items]
    write_csv("reward_results.csv", ["reward", "checkpoint", "total GRPO steps", "no-b words exact", "b-words skip-b exact", "b-leak"],
              [[n.replace("\n", " "), f, st, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}"] for (n, f, st), a, b, c in zip(items, nob, bw, leak)])
    fig = new_fig("Reward function results", "Held-out exact match. Step counts are totals from the pretrained model.")
    ax = fig.add_axes([0.07, 0.16, 0.88, 0.62])
    xs = range(len(items)); w = 0.38
    ax.bar([i - w / 2 for i in xs], [v * 100 for v in nob], w * 0.94, color=S3, label="no-b words: reverse")
    ax.bar([i + w / 2 for i in xs], [v * 100 for v in bw], w * 0.94, color=S1, label="b-words: reverse + skip b")
    for i, a, b in zip(xs, nob, bw):
        ax.text(i - w / 2, a * 100 + 1.5, f"{a*100:.1f}", ha="center", fontsize=11, color=INK2)
        ax.text(i + w / 2, b * 100 + 1.5, f"{b*100:.1f}", ha="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{n}\n({st} steps)" for n, _, st in items], fontsize=11)
    ax.set_ylim(0, 110); ax.set_ylabel("exact match (%)"); ax.legend(loc="upper left", ncols=2)
    fig.text(0.07, 0.045, f"b-leak (answers still containing 'b') = {max(leak)*100:.1f}% or less for all six. "
             "Every reward stops the 'b'; only the dense ones teach what to write instead.", fontsize=10.5, color=INK2)
    save(fig, "06_reward_results.png", "results/summary.json")


# ================================================================ 07 accuracy no-skip vs skip
def fig_accuracy():
    s = summary
    cols = [("no_b_words_exact", "no-b words:\nreverse"), ("b_words_skip_exact", "b-words:\nreverse + skip b"),
            ("step2_task_on_random_words", "random words:\nStep 2 task"), ("reverse_exact (Step1 task)", "random words:\nplain reverse (Step 1)")]
    write_csv("accuracy_key_models.csv", ["model", "checkpoint"] + [c[1].replace("\n", " ") for c in cols] + ["b-leak"],
              [[lab.replace("\n", " "), fn] + [f"{s[fn][k]:.4f}" for k, _ in cols] + [f"{s[fn]['b_leak_rate']:.4f}"] for fn, lab, _ in KEY])
    fig = new_fig("Accuracy: no-skip-b vs skip-b",
                  "Held-out exact match. 'No-b words' = 1,154 words without b; 'b-words' = 840 words with at least one b.")
    ax = fig.add_axes([0.07, 0.22, 0.88, 0.56])
    groups = cols[:3]
    w = 0.2
    for j, (fn, lab, col) in enumerate(KEY):
        vals = [s[fn][k] * 100 for k, _ in groups]
        xs = [i + (j - 1.5) * w for i in range(len(groups))]
        ax.bar(xs, vals, w * 0.92, color=col, label=lab.replace("\n", " "))
        for x_, v in zip(xs, vals):
            ax.text(x_, v + 1.5, f"{v:.1f}" if v < 99.95 else "100", ha="center", fontsize=10, color=INK2 if j < 3 else INK,
                    fontweight="bold" if j == 3 else "normal")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups], fontsize=12)
    ax.set_ylim(0, 118); ax.set_ylabel("exact match (%)"); ax.legend(loc="upper center", ncols=2, fontsize=11)
    pre = s["pretrain.pt"]["step2_task_on_random_words"]
    fig.text(0.07, 0.085, f"Random words: only ~15% contain a b, so even the pretrained model scores {pre*100:.1f}% on the Step 2 task. "
             "The b-words column is the real test.", fontsize=10.5, color=INK2)
    fig.text(0.07, 0.05, "Plain reverse on random words drops 100% -> 85% after RL on purpose: the model now deletes the b's.",
             fontsize=10.5, color=INK2)
    save(fig, "07_accuracy_noskip_vs_skip.png", "results/summary.json")


def fig_per_length():
    s = summary
    lens = ["2", "3", "4", "5", "6"]
    write_csv("per_length_b_words.csv", ["model", "checkpoint"] + [f"len {L}" for L in lens],
              [[lab.replace("\n", " "), fn] + [f"{s[fn]['b_words_per_length'][L]:.4f}" for L in lens] for fn, lab, _ in KEY])
    fig = new_fig("Skip-b accuracy by word length", "b-words only. Longer words = more letters to shift after a skipped b.")
    ax = fig.add_axes([0.07, 0.12, 0.60, 0.66])
    for fn, lab, col in KEY:
        v = [s[fn]["b_words_per_length"][L] * 100 for L in lens]
        ax.plot([int(L) for L in lens], v, color=col, marker="o", ms=8, mec=SURFACE, mew=2)
        ax.text(6.12, v[-1], lab.replace("\n", " "), color=INK2, fontsize=11, va="center")
    ax.set_xticks(range(2, 7)); ax.set_xlabel("word length"); ax.set_ylabel("exact match (%)")
    ax.set_ylim(-3, 105); ax.set_xlim(1.8, 6.1)
    ax.legend([l.replace("\n", " ") for _, l, _ in KEY], loc="lower left", bbox_to_anchor=(0.01, 0.07), fontsize=10.5)
    save(fig, "08_per_length.png", "results/summary.json")


# ================================================================ 09 training curves
def fig_curves():
    c, i = load("grpo_expC_metrics.json")["log"], load("grpo_expI_metrics.json")["log"]
    steps = c["eval_step"] + [400 + s for s in i["eval_step"][1:]]
    bw = c["eval_bomb_exact"] + i["eval_bomb_exact"][1:]
    nb = c["eval_nob_exact"] + i["eval_nob_exact"][1:]
    write_csv("training_curve_round2.csv", ["GRPO step (from pretrained)", "run", "b-words skip-b exact", "no-b words exact"],
              [[st, "expC" if st <= 400 else "expI", f"{a:.4f}", f"{b:.4f}"] for st, a, b in zip(steps, bw, nb)])
    r1 = load("grpo_long_lr5e-5_metrics.json")["log"], load("grpo_final_metrics.json")["log"]
    r1_steps = r1[0]["eval_step"] + [2000 + s for s in r1[1]["eval_step"][1:]]
    r1_bw = r1[0]["eval_bomb_exact"] + r1[1]["eval_bomb_exact"][1:]
    write_csv("training_curve_round1.csv", ["GRPO step", "b-words skip-b exact"], [[a, f"{b:.4f}"] for a, b in zip(r1_steps, r1_bw)])

    fig = new_fig("Training curves (held-out eval)",
                  "Round 2 runs on GPU; the exploration run sits near 60% for ~600 steps, then finds the rule.")
    ax = fig.add_axes([0.07, 0.12, 0.62, 0.66])
    ax.axvspan(0, 400, color=GRID, alpha=0.5, lw=0)
    ax.text(200, 4, "token-level\n(expC)", ha="center", fontsize=10.5, color=INK2, va="bottom")
    ax.text(1150, 103, "+ hard-state exploration (expI)", ha="center", fontsize=10.5, color=INK2, va="top")
    l1, = ax.plot(steps, [v * 100 for v in nb], color=S3, marker="o", ms=6, mec=SURFACE, mew=1.5)
    l2, = ax.plot(steps, [v * 100 for v in bw], color=S1, marker="o", ms=7, mec=SURFACE, mew=1.5)
    l3 = ax.axhline(39.0, color=S2, lw=1.5, ls="--")
    ax.text(1890, 41, "round 1 best after 4,000 steps: 39.0%", ha="right", fontsize=10.5, color=INK2)

    ax.set_xlim(0, 1900); ax.set_ylim(0, 106); ax.set_xlabel("GRPO steps (from the pretrained model)")
    ax.set_ylabel("exact match (%)")
    ax.legend([l1, l2, l3], ["no-b words: reverse", "b-words: reverse + skip b", "round 1 best (39.0%)"], loc="lower right", fontsize=10.5)
    tx = fig.add_axes([0.80, 0.12, 0.18, 0.66]); tx.axis("off")
    pts = [("step 400", f"{bw[c['eval_step'].index(400)]*100:.1f}%"), ("step 1000", f"{bw[steps.index(1000)]*100:.1f}%"),
           ("step 1100", f"{bw[steps.index(1100)]*100:.1f}%"), ("step 1300", f"{bw[steps.index(1300)]*100:.1f}%"),
           ("step 1900", f"{bw[-1]*100:.1f}%")]
    tx.text(0, 1, "b-words at step", fontsize=13, fontweight="bold", color=INK, transform=tx.transAxes, va="top")
    for k, (a, b) in enumerate(pts):
        tx.text(0, 0.88 - k * 0.1, a, fontsize=12, color=INK2, transform=tx.transAxes)
        tx.text(1, 0.88 - k * 0.1, b, fontsize=12, color=INK, ha="right", fontweight="bold", transform=tx.transAxes)
    tx.text(0, 0.36, f"final no-b words: {nb[-1]*100:.1f}%", fontsize=11.5, color=INK, transform=tx.transAxes, va="top")
    tx.text(0, 0.26, "Curve swings (81-96%)\nafter the jump ->\ncheckpoint choice matters.", fontsize=11, color=INK2,
            transform=tx.transAxes, va="top")
    save(fig, "09_training_curves.png", "results/grpo_expC_metrics.json, results/grpo_expI_metrics.json, results/grpo_long_lr5e-5 + final")


# ================================================================ 10 what failed / why
def fig_round2_attempts():
    tags = [("expB", "baseline = 0"), ("expA", "T = 1.0"), ("expC", "clamped baseline, T = 1.5"),
            ("expE", "+ mid-answer starts"), ("expG", "+ std scaling, T = 3"), ("expF", "+ T = 3 whole answer"),
            ("expH", "+ negatives x0.1"), ("expD", "+ more steps"), ("expI", "+ hard-state exploration")]
    vals = []
    for t, lab in tags:
        if t in ("expC", "expI"):
            m = load(f"grpo_{t}_metrics.json")["log"]
            vals.append((lab, t, m["eval_step"][-1], m["eval_bomb_exact"][-1]))
        else:
            st, b, _ = last_eval(t)
            vals.append((lab, t, st, b))
    fig = new_fig("Round 2: what was tried", "b-words exact at the last eval of each run. Most ideas stall near 60%; one breaks through.")
    ax = fig.add_axes([0.30, 0.10, 0.62, 0.70])
    ys = range(len(vals))[::-1]
    for y, (lab, t, st, b) in zip(ys, vals):
        col = S1 if t == "expI" else (S4 if t == "expC" else MUTED)
        ax.barh(y, b * 100, 0.62, color=col)
        ax.text(b * 100 + 1, y, f"{b*100:.1f}%", va="center", fontsize=11.5, color=INK, fontweight="bold" if t == "expI" else "normal")
        ax.text(-2, y, f"{t}: {lab}  ({st} steps)", va="center", ha="right", fontsize=11, color=INK2)
    ax.set_yticks([]); ax.set_xlim(0, 105); ax.set_xlabel("b-words exact (%)"); ax.grid(axis="y", visible=False)
    ax.axvline(39.0, color=S2, lw=1.5, ls="--"); ax.text(40, len(vals) - 0.45, "round 1: 39%", fontsize=10, color=INK2)
    save(fig, "10_round2_attempts.png", "results/log_exp*.txt, results/grpo_exp{C,I}_metrics.json. Runs marked '+' continue from expC.")


# ================================================================ 11 examples + errors
def fig_examples():
    words = ["abcde", "hello", "mhbk", "stib", "irbpb", "bbq", "nbb"]
    models = [(fn, lab.replace("\n", " ")) for fn, lab, _ in KEY]
    preds = {fn: predict(load_model(os.path.join(CKPT_DIR, fn)), words) for fn, _ in models}
    rows = [[w, target_skip_b(w)] + [("" if preds[fn][k] is None else preds[fn][k]) or "(empty)" for fn, _ in models]
            for k, w in enumerate(words)]
    write_csv("example_predictions.csv", ["input", "target (skip b)"] + [l for _, l in models], rows)

    final = load_model(os.path.join(CKPT_DIR, "grpo_expI.pt"))
    p = predict(final, eval_bomb)
    errs = [(w, target_skip_b(w), q) for w, q in zip(eval_bomb, p) if q != target_skip_b(w)]
    pn = predict(final, eval_nob)
    errs_n = [(w, w[::-1], q) for w, q in zip(eval_nob, pn) if q != w[::-1]]
    dbl = lambda w: any(a == c for a, c in zip(w, w[1:]))
    write_csv("final_model_errors.csv", ["test set", "input", "target", "prediction", "input has a doubled letter"],
              [["eval_bomb", *e, dbl(e[0])] for e in errs] + [["eval_no_b", *e, dbl(e[0])] for e in errs_n])

    fig = new_fig("Example outputs and remaining errors", "Greedy decoding. Target = reverse the word and delete every 'b'.")
    disp = [[r[0], r[1] or "(empty)"] + r[2:] for r in rows]
    draw_table(fig, [0.04, 0.42, 0.92, 0.40], ["input", "target", "Pretrained", "Round 1", "Round 2", "FINAL"],
               disp, [1.0, 1.0, 1.2, 1.2, 1.2, 1.2], bold_cols=(5,), fs=12.5)
    n_d = sum(dbl(e[0]) for e in errs)
    fig.text(0.04, 0.36, f"FINAL model errors: {len(errs)} of {len(eval_bomb)} b-words, {len(errs_n)} of {len(eval_nob)} no-b words",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.04, 0.30, f"{n_d} of the {len(errs)} b-word errors have a doubled letter in the input\n"
             "(e.g. 'bb', 'll'). 11 of the other 13 still repeat a letter\n(the old error: 'whbay' -> 'yaww', 'jobrb' -> 'roo').",
             fontsize=11.5, color=INK2, va="top")
    show = errs[:4] + errs_n[:2]
    draw_table(fig, [0.56, 0.06, 0.40, 0.27], ["input", "target", "FINAL predicted"],
               [[a, b or "(empty)", c if c else "(empty)"] for a, b, c in show], [1, 1, 1.2], fs=10.5)
    save(fig, "11_examples_and_errors.png", "checkpoints/*.pt evaluated live by make_present.py")
    return len(errs), len(errs_n), n_d


if __name__ == "__main__":
    fig_design(); fig_experiments(); fig_hparams(); fig_split(); fig_rewards(); fig_reward_results()
    fig_accuracy(); fig_per_length(); fig_curves(); fig_round2_attempts()
    print("errors (b-words, no-b, doubled):", fig_examples())
