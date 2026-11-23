"""
Step 2 — Fine-tune the pretrained MiniGPT with GRPO (Group Relative Policy Optimization)
so that it SKIPS the letter "b" (bomb) in the reversed output.
    <BOS> a b c d e <SEP>  ->  e d c a <EOS>

No labels are given to the model. It only receives a REWARD for what it generates.

Run:  .venv\\Scripts\\python.exe train_grpo.py                 (main reward)
      .venv\\Scripts\\python.exe train_grpo.py --reward naive  (ablation: "no b" reward only)
Out:  checkpoints/grpo_<reward>.pt, results/grpo_<reward>_metrics.json, results/grpo_<reward>_curves.png
"""
import argparse
import copy
import json
import os
import time
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from common import (BLOCK_SIZE, EOS, CKPT_DIR, RESULTS_DIR, WordSampler, build_eval_sets,
                    evaluate, generate, load_model, parse_completion, save_model, set_seed,
                    target_reverse, target_skip_b)

# ------------------------- reward functions -------------------------
def common_prefix_len(a, b):
    n = 0
    for c1, c2 in zip(a, b):
        if c1 != c2:
            break
        n += 1
    return n


def reward_shaped(x, out):
    """Main reward. x = input word, out = generated answer (str) or None if malformed.

      * malformed output (no <EOS> / special token in answer) ...... -1.0
      * correct-prefix score: (# leading chars that are right)
                              / max(len(answer), len(target)) .... 0 .. +1
      * every 'b' written in the output ............................ -0.5 each
      * wrong length: -0.5 * |len(out) - len(target)| / len(target)  (stops "quit early")
      * exactly correct answer ...................................... +1.0 bonus
    Range: about [-4, 2].

    Why a *prefix* score?  It rewards progress token by token:
      target 'edca'   'edcba' -> 0.6-0.5-0.125 = -0.03 (wrote b)
                      'edcxa' -> 0.6-0.125     =  0.48 (replaced b)
                      'edc'   -> 0.75-0.125    =  0.63 (stopped at b)
                      'edcaa' -> 0.8-0.125     =  0.68 (skipped b, but too long)
                      'edca'  -> 1.0+1.0       =  2.00 (perfect)
    so each small step towards the right behaviour gets a higher reward.
    """
    if out is None:
        return -1.0
    tgt = target_skip_b(x)
    denom = max(len(out), len(tgt))
    r = common_prefix_len(out, tgt) / denom if denom > 0 else 1.0
    r -= 0.5 * out.count("b")
    r -= 0.5 * abs(len(out) - len(tgt)) / max(len(tgt), 1)
    if out == tgt:
        r += 1.0
    return r


def reward_similarity(x, out):
    """Ablation: SequenceMatcher similarity instead of prefix (first version we tried)."""
    if out is None:
        return -1.0
    tgt = target_skip_b(x)
    r = SequenceMatcher(None, out, tgt).ratio()
    r -= 0.5 * out.count("b")
    if out == tgt:
        r += 1.0
    return r


def reward_naive(x, out):
    """Ablation: only says 'do not write b'. Does NOT say what the answer should be."""
    if out is None:
        return -1.0
    return 1.0 if "b" not in out else 0.0


REWARDS = {"shaped": reward_shaped, "similarity": reward_similarity, "naive": reward_naive}


# ------------------------- GRPO helpers -------------------------
def completion_logprobs(model, seq, plen, temperature, return_entropy=False):
    """Log-prob of every generated token + a mask that marks completion tokens
    (from the first answer token up to and including <EOS>)."""
    logits, _ = model(seq[:, :-1])
    logp_all = F.log_softmax(logits / temperature, dim=-1)
    tgt = seq[:, 1:]
    logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # [B, T-1]
    entropy = -(logp_all.exp() * logp_all).sum(-1)                 # [B, T-1]
    B, T = tgt.shape
    mask = torch.zeros(B, T, dtype=torch.float32, device=seq.device)
    for i in range(B):
        start = plen[i] - 1                    # position whose next token is the 1st answer token
        row = seq[i, plen[i]:].tolist()
        n = len(row)
        if EOS in row:
            n = row.index(EOS) + 1
        mask[i, start: start + n] = 1.0
    if return_entropy:
        return logp, mask, entropy
    return logp, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="shaped", choices=list(REWARDS))
    ap.add_argument("--init", default="pretrain", help="checkpoint name to start from")
    ap.add_argument("--ref", default=None, help="reference model for KL (default = --init)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default=None, help="name for outputs (default = reward name)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts", type=int, default=32, help="prompts per step")
    ap.add_argument("--group", type=int, default=16, help="G = samples per prompt")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.02, help="KL penalty weight")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO clip epsilon")
    ap.add_argument("--inner", type=int, default=2, help="gradient updates per batch")
    ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--ent", type=float, default=0.0, help="entropy bonus weight")
    ap.add_argument("--b_frac", type=float, default=0.6, help="fraction of prompts containing b")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cpu"
    reward_fn = REWARDS[args.reward]

    eval_rev, eval_bomb = build_eval_sets()
    eval_nob = [w for w in eval_rev if "b" not in w]          # words without b
    sampler = WordSampler(held_out=eval_rev + eval_bomb, seed=1000 + args.seed)

    policy = load_model(os.path.join(CKPT_DIR, f"{args.init}.pt"), device)
    if args.ref:
        ref = load_model(os.path.join(CKPT_DIR, f"{args.ref}.pt"), device).eval()
    else:
        ref = copy.deepcopy(policy).eval()                     # frozen reference model
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    def run_eval():
        rb = evaluate(policy, eval_bomb, target_skip_b, device)
        rn = evaluate(policy, eval_nob, target_reverse, device)
        return rb, rn

    log = {k: [] for k in ["step", "reward", "reward_std", "exact_rollout", "b_rate_rollout",
                           "kl", "eval_step", "eval_bomb_exact", "eval_nob_exact", "eval_b_leak"]}
    t0 = time.time()
    G = args.group
    for step in range(args.steps + 1):
        if step % 25 == 0:
            rb, rn = run_eval()
            log["eval_step"].append(step)
            log["eval_bomb_exact"].append(rb["exact_match"])
            log["eval_nob_exact"].append(rn["exact_match"])
            log["eval_b_leak"].append(rb["b_leak_rate"])
            print(f"[eval] step {step:4d} | bomb-words exact {rb['exact_match']*100:6.2f}% "
                  f"| b-leak {rb['b_leak_rate']*100:6.2f}% | no-b words exact {rn['exact_match']*100:6.2f}% "
                  f"| {time.time()-t0:5.1f}s")
        if step == args.steps:
            break

        # ---- 1) sample a batch of prompts, each repeated G times ----
        words = [sampler.sample(must_have_b=(i < int(args.b_frac * args.prompts)))
                 for i in range(args.prompts)]
        rep = [w for w in words for _ in range(G)]
        seq, plen = generate(policy, rep, greedy=False, temperature=args.temp, device=device)
        outs = [parse_completion(seq[i], plen[i]) for i in range(len(rep))]

        # ---- 2) rewards and group-relative advantages ----
        R = torch.tensor([reward_fn(w, o) for w, o in zip(rep, outs)], dtype=torch.float32)
        Rg = R.view(args.prompts, G)
        adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)).view(-1)

        with torch.no_grad():
            old_logp, mask = completion_logprobs(policy, seq, plen, args.temp)
            ref_logp, _ = completion_logprobs(ref, seq, plen, args.temp)

        # ---- 3) clipped policy-gradient update (PPO-style) + KL to reference ----
        policy.train()
        for _ in range(args.inner):
            logp, _, ent = completion_logprobs(policy, seq, plen, args.temp, return_entropy=True)
            ratio = torch.exp(logp - old_logp)
            a = adv.unsqueeze(1)
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a)
            # k3 estimator of KL(policy || ref), always >= 0
            d = ref_logp - logp
            kl = torch.exp(d) - d - 1
            # entropy bonus keeps the policy exploring (prevents "entropy collapse")
            per_tok = -(surr - args.beta * kl + args.ent * ent)
            loss = ((per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if step % 5 == 0:
            exact = sum(o == target_skip_b(w) for w, o in zip(rep, outs)) / len(rep)
            b_rate = sum((o is not None and "b" in o) for o in outs) / len(rep)
            kl_mean = ((kl * mask).sum() / mask.sum()).item()
            log["step"].append(step); log["reward"].append(R.mean().item())
            log["reward_std"].append(Rg.std(1).mean().item())
            log["exact_rollout"].append(exact); log["b_rate_rollout"].append(b_rate)
            log["kl"].append(kl_mean)
            if step % 25 == 0:
                print(f"  step {step:4d} | reward {R.mean():+.3f} | group-std {Rg.std(1).mean():.3f} "
                      f"| rollout exact {exact*100:5.1f}% | rollout b {b_rate*100:5.1f}% | KL {kl_mean:.4f}")

    tag = args.tag or args.reward
    save_model(policy, os.path.join(CKPT_DIR, f"grpo_{tag}.pt"), extra=vars(args))
    rb, rn = run_eval()
    out = {"args": vars(args), "time_sec": time.time() - t0,
           "eval_bomb_skip_b": rb, "eval_no_b_reverse": rn, "log": log}
    json.dump(out, open(os.path.join(RESULTS_DIR, f"grpo_{tag}_metrics.json"), "w"), indent=2)
    print("examples:")
    for e in rb["examples"][:8]:
        print("  ", e)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        ax[0].plot(log["step"], log["reward"]); ax[0].set_title(f"Mean reward ({tag})")
        ax[1].plot(log["eval_step"], [v * 100 for v in log["eval_bomb_exact"]], marker="o", label="bomb words: exact (skip b)")
        ax[1].plot(log["eval_step"], [v * 100 for v in log["eval_nob_exact"]], marker="s", label="no-b words: exact")
        ax[1].plot(log["eval_step"], [v * 100 for v in log["eval_b_leak"]], marker="^", label="b-leak rate")
        ax[1].set_ylim(-2, 102); ax[1].legend(fontsize=8); ax[1].set_title("Held-out eval (%)")
        ax[2].plot(log["step"], log["kl"]); ax[2].set_title("KL(policy || ref)")
        for a_ in ax: a_.set_xlabel("GRPO step")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"grpo_{tag}_curves.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
