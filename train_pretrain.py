"""
Step 1 — Train MiniGPT from scratch to reverse a string.
    <BOS> h e l l o <SEP>  ->  o l l e h <EOS>

Run:  .venv\\Scripts\\python.exe train_pretrain.py
Out:  checkpoints/pretrain.pt, results/pretrain_metrics.json, results/pretrain_loss.png
"""
import json
import math
import os
import time

import torch

from common import (GPTConfig, MiniGPT, WordSampler, build_eval_sets, evaluate,
                    make_sft_batch, save_model, set_seed, target_reverse, target_skip_b,
                    CKPT_DIR, RESULTS_DIR)

# ------------------------- hyper-parameters -------------------------
SEED = 42
STEPS = 3000
BATCH_SIZE = 256
LR = 1e-3
WARMUP = 200
WEIGHT_DECAY = 0.01
EVAL_EVERY = 250
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lr_at(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR * 0.1 + 0.9 * LR * 0.5 * (1 + math.cos(math.pi * prog))   # cosine to 10%


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_smoothing", type=float, default=0.1,
                    help="0.1 keeps the model a little 'unsure' -> RL can explore later")
    ap.add_argument("--name", default="pretrain")
    args = ap.parse_args()
    set_seed(SEED)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    eval_rev, eval_bomb = build_eval_sets()
    sampler = WordSampler(held_out=eval_rev + eval_bomb, seed=SEED)

    cfg = GPTConfig()
    model = MiniGPT(cfg).to(DEVICE)
    print(f"MiniGPT config: {cfg}")
    print(f"parameters: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.98))
    log = {"step": [], "loss": [], "eval_step": [], "eval_exact": []}
    t0 = time.time()
    for step in range(STEPS):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        words = [sampler.sample() for _ in range(BATCH_SIZE)]
        X, Y = make_sft_batch(words, target_reverse, DEVICE)
        logits, _ = model(X)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100,
            label_smoothing=args.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 25 == 0:
            log["step"].append(step); log["loss"].append(loss.item())
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            r = evaluate(model, eval_rev, target_reverse, DEVICE)
            log["eval_step"].append(step); log["eval_exact"].append(r["exact_match"])
            print(f"step {step:5d} | loss {loss.item():.4f} | eval exact {r['exact_match']*100:6.2f}%"
                  f" | {time.time()-t0:5.1f}s")

    save_model(model, os.path.join(CKPT_DIR, f"{args.name}.pt"),
               extra={"label_smoothing": args.label_smoothing})

    # ---------------- final evaluation ----------------
    res_rev = evaluate(model, eval_rev, target_reverse, DEVICE)
    # the pretrained model on the Step-2 task (it should still write the 'b')
    res_bomb_task = evaluate(model, eval_bomb, target_skip_b, DEVICE)
    res_bomb_rev = evaluate(model, eval_bomb, target_reverse, DEVICE)
    out = {
        "config": cfg.__dict__, "params": model.num_params(),
        "train": {"label_smoothing": args.label_smoothing, "steps": STEPS, "batch_size": BATCH_SIZE, "lr": LR, "warmup": WARMUP,
                  "weight_decay": WEIGHT_DECAY, "examples_seen": STEPS * BATCH_SIZE,
                  "time_sec": time.time() - t0},
        "eval_reverse": res_rev,
        "eval_bomb_reverse_task": res_bomb_rev,
        "eval_bomb_skip_b_task": res_bomb_task,
        "log": log,
    }
    json.dump(out, open(os.path.join(RESULTS_DIR, f"{args.name}_metrics.json"), "w"), indent=2)
    print(json.dumps({k: out[k]["exact_match"] for k in
                      ["eval_reverse", "eval_bomb_reverse_task", "eval_bomb_skip_b_task"]}, indent=2))

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(log["step"], log["loss"]); ax[0].set_yscale("log")
        ax[0].set_title("Pretraining loss (answer tokens)"); ax[0].set_xlabel("step")
        ax[1].plot(log["eval_step"], [v * 100 for v in log["eval_exact"]], marker="o")
        ax[1].set_title("Held-out exact match (%)"); ax[1].set_xlabel("step"); ax[1].set_ylim(0, 101)
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"{args.name}_curves.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
