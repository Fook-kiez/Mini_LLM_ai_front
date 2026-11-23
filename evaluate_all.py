"""
Evaluate every checkpoint on the same held-out sets and write a comparison table.

Run:  .venv\\Scripts\\python.exe evaluate_all.py
Out:  results/summary.md, results/summary.json, results/summary.png
"""
import json
import os
import sys

import torch

from common import (CKPT_DIR, RESULTS_DIR, build_eval_sets, evaluate, load_model,
                    target_reverse, target_skip_b)

MODELS = [  # (checkpoint file, label)
    ("pretrain_nols.pt", "Pretrained (no label smoothing)"),
    ("pretrain.pt", "Pretrained (label smoothing 0.1)"),
    ("grpo_naive.pt", "GRPO - naive reward"),
    ("grpo_similarity.pt", "GRPO - similarity reward"),
    ("grpo_nols.pt", "GRPO - shaped reward, from no-LS model"),
    ("grpo_shaped.pt", "GRPO - shaped reward, 600 steps"),
    ("grpo_ent02.pt", "GRPO - shaped + entropy 0.02"),
    ("grpo_ent05.pt", "GRPO - shaped + entropy 0.05"),
    ("grpo_long_g32.pt", "GRPO - shaped, G=32, 2000 steps"),
    ("grpo_long_lr5e-5.pt", "GRPO - shaped, G=16, 2000 steps"),
    ("grpo_final.pt", "GRPO - shaped, 4000 steps (FINAL)"),
]


def main():
    torch.set_num_threads(8)
    eval_rev, eval_bomb = build_eval_sets()
    eval_nob = [w for w in eval_rev if "b" not in w]
    rows = []
    for fn, label in MODELS:
        path = os.path.join(CKPT_DIR, fn)
        if not os.path.exists(path):
            continue
        m = load_model(path)
        r_rev = evaluate(m, eval_rev, target_reverse)
        r_nob = evaluate(m, eval_nob, target_reverse)
        r_bomb = evaluate(m, eval_bomb, target_skip_b)
        # overall Step-2 task on a natural mix of words (target = reverse without b)
        r_mix = evaluate(m, eval_rev, target_skip_b)
        rows.append({
            "model": label, "file": fn,
            "reverse_exact (Step1 task)": r_rev["exact_match"],
            "no_b_words_exact": r_nob["exact_match"],
            "b_words_skip_exact": r_bomb["exact_match"],
            "b_words_char_acc": r_bomb["char_accuracy"],
            "b_leak_rate": r_bomb["b_leak_rate"],
            "invalid_rate": r_bomb["invalid_rate"],
            "step2_task_on_random_words": r_mix["exact_match"],
            "b_words_per_length": r_bomb["per_length"],
            "examples": r_bomb["examples"][:6],
        })
        print(f"{label:45s} rev {r_rev['exact_match']*100:6.2f} | no-b {r_nob['exact_match']*100:6.2f} "
              f"| b-skip {r_bomb['exact_match']*100:6.2f} | leak {r_bomb['b_leak_rate']*100:6.2f}")

    json.dump(rows, open(os.path.join(RESULTS_DIR, "summary.json"), "w"), indent=2)
    pct = lambda v: f"{v*100:.1f}%"
    lines = ["| Model | Reverse (Step 1) | No-b words | b-words skip-b exact | b-leak | Step-2 task (random words) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {pct(r['reverse_exact (Step1 task)'])} | {pct(r['no_b_words_exact'])} | "
                     f"{pct(r['b_words_skip_exact'])} | {pct(r['b_leak_rate'])} | {pct(r['step2_task_on_random_words'])} |")
    lines += ["", "Per-length exact match on b-words (skip-b task):", "",
              "| Model | " + " | ".join(f"len {L}" for L in range(2, 7)) + " |",
              "|---|" + "---|" * 5]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(pct(r['b_words_per_length'][str(L)]) for L in range(2, 7)) + " |")
    open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [r["model"].replace(" - ", "\n").replace(" (", "\n(").replace(", ", ",\n") for r in rows]
        keys = [("no_b_words_exact", "no-b words: reverse exact"),
                ("b_words_skip_exact", "b-words: skip-b exact"),
                ("b_leak_rate", "b-leak rate")]
        x = np.arange(len(rows)); w = 0.27
        fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(rows)), 4.2))
        for i, (k, name) in enumerate(keys):
            ax.bar(x + (i - 1) * w, [r[k] * 100 for r in rows], w, label=name)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(fontsize=8)
        ax.set_title("Held-out evaluation of all models")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "summary.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
