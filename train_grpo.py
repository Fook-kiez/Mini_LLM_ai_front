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

from common import (BLOCK_SIZE, EOS, PAD, MAX_NEW_TOKENS, full_ids, CKPT_DIR, RESULTS_DIR, WordSampler, build_eval_sets,
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
def completion_logprobs(model, seq, plen, temperature, return_entropy=False, max_new=None):
    """Log-prob of every generated token + a mask that marks completion tokens
    (from the first answer token up to and including <EOS>)."""
    logits, _ = model(seq[:, :-1])
    if torch.is_tensor(temperature):
        temperature = temperature.view(-1, 1, 1)          # one temperature per row
    logp_all = F.log_softmax(logits / temperature, dim=-1)
    tgt = seq[:, 1:]
    logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # [B, T-1]
    entropy = -(logp_all.exp() * logp_all).sum(-1)                 # [B, T-1]
    mask = completion_mask(seq, plen, max_new)
    if return_entropy:
        return logp, mask, entropy
    return logp, mask


def completion_mask(seq, plen, max_new=None):
    """[B, T-1] float mask over logit positions whose next token is a generated token
    (first answer token .. <EOS> inclusive). Vectorised (no per-row GPU syncs)."""
    B, T = seq.shape
    pl = torch.as_tensor(plen, device=seq.device).unsqueeze(1)          # [B,1]
    pos = torch.arange(T, device=seq.device).unsqueeze(0)               # [1,T]
    in_comp = pos >= pl
    is_eos = (seq == EOS) & in_comp
    # index of first EOS in the completion, else last position generate() could write
    first_eos = torch.where(is_eos, pos, torch.full_like(pos, T)).min(1, keepdim=True).values
    mn = MAX_NEW_TOKENS if max_new is None else max_new.view(-1, 1)
    last = torch.minimum(first_eos, pl + mn - 1).clamp(max=T - 1)
    tok_mask = in_comp & (pos <= last)                                  # over seq positions
    return tok_mask[:, 1:].float()                                      # shift: logit j -> token j+1


def target_tensor(words, T, device):
    """Ideal full sequence <BOS> x <SEP> skip_b(reverse(x)) <EOS>, padded to T."""
    tg = torch.full((len(words), T), PAD, dtype=torch.long)
    for i, w in enumerate(words):
        ids = full_ids(w, target_skip_b(w))
        tg[i, :len(ids)] = torch.tensor(ids)
    return tg.to(device)


def token_advantages(seq, plen, words, G, mask, gamma=0.0, baseline="mixed", neg_scale=1.0):
    """Token-level GRPO advantage ("process reward" + per-state group baseline).

    Per generated token k (a verifier checks it, the model never sees the label):
        r_k = +1  if the answer so far (incl. token k) is still exactly on track
        r_k = -1  if token k is the FIRST mistake (wrote 'b', repeated a letter, stopped early ...)
        r_k =  0  after the first mistake (those tokens are ignored: advantage 0)
    Return  G_k = sum_j gamma^(j-k) r_j.

    Baseline: all G samples of one prompt that are still on track at position k are in
    exactly the SAME state (same prompt + same prefix), so their mean return is an
    unbiased value estimate for that state -> A_k = G_k - mean_group_state(G_k).
    This is GRPO's group baseline applied per state instead of per whole sequence,
    so the token right after a skipped 'b' gets its own credit signal.
    """
    B, Tm1 = mask.shape
    tgt = target_tensor(words, seq.shape[1], seq.device)
    pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
    pl = torch.as_tensor(plen, device=seq.device).unsqueeze(1)
    gen = torch.cat([torch.zeros(B, 1, device=seq.device), mask], 1).bool()   # seq positions
    correct = (seq == tgt) | ~gen | (pos < pl)          # non-generated positions count as ok
    on_track = torch.cumprod(correct.long(), 1).bool()  # prefix up to & incl. p is correct
    prev_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=seq.device), on_track[:, :-1]], 1)
    active = gen & prev_ok                              # decision made from an on-track state
    r = torch.where(on_track, 1.0, -1.0) * active.float()
    ret = torch.zeros_like(r)
    run = torch.zeros(B, device=seq.device)
    for p in range(seq.shape[1] - 1, -1, -1):
        run = r[:, p] + gamma * run
        ret[:, p] = run
    ret = ret * active.float()
    a = active.float().view(-1, G, seq.shape[1])
    rg = ret.view(-1, G, seq.shape[1])
    base = (rg * a).sum(1, keepdim=True) / a.sum(1, keepdim=True).clamp(min=1)
    if baseline == "none":
        base = torch.zeros_like(base)
    elif baseline in ("mixed", "mixed_std"):
        # If EVERY sample in a state makes the same mistake, the pure group baseline gives
        # advantage 0 and the model can never escape (the correct token has p~1e-4 and is
        # never sampled). Clamping the baseline at 0 keeps a "push the mistake down" signal
        # for those states while still centring states where some samples succeed.
        base = base.clamp(min=0.0)
    adv = (rg - base) * a
    if baseline == "mixed_std":
        # GRPO-style per-state scaling: a rare success (1 of 16) gets a ~4x larger push
        n = a.sum(1, keepdim=True)
        var = (((rg - (rg * a).sum(1, keepdim=True) / n.clamp(min=1)) ** 2) * a).sum(1, keepdim=True) / n.clamp(min=1)
        adv = adv / var.sqrt().clamp(min=0.25)
    # asymmetric update: at high sampling temperature most sampled tokens are random
    # mistakes; full-strength "push down" on all of them drowns the rare successes.
    adv = torch.where(adv < 0, adv * neg_scale, adv)
    adv = adv.reshape(B, -1)
    # no std-normalisation: most on-track tokens have advantage 0, so a batch std would be
    # tiny and blow up the step size. Raw advantages are already bounded in [-2, 2].
    return adv[:, 1:], active[:, 1:].float()            # align with logit positions


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
    ap.add_argument("--adv", default="seq", choices=["seq", "token"],
                    help="seq = classic GRPO (one advantage per answer); "
                         "token = per-token process reward with per-state group baseline")
    ap.add_argument("--gamma", type=float, default=0.0, help="discount for --adv token")
    ap.add_argument("--baseline", default="mixed", choices=["state", "mixed", "mixed_std", "none"],
                    help="baseline for --adv token (see token_advantages)")
    ap.add_argument("--neg_scale", type=float, default=1.0,
                    help="multiply negative token advantages by this (asymmetric PG)")
    ap.add_argument("--explore_frac", type=float, default=0.0,
                    help="fraction of prompts used as one-step exploration episodes from hard states")
    ap.add_argument("--explore_temp", type=float, default=3.0)
    ap.add_argument("--reset_frac", type=float, default=0.0,
                    help="fraction of prompts whose episode starts from a random correct prefix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval_every", type=int, default=25)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
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
    best_score = -1.0
    G = args.group
    for step in range(args.steps + 1):
        if step % args.eval_every == 0:
            rb, rn = run_eval()
            log["eval_step"].append(step)
            log["eval_bomb_exact"].append(rb["exact_match"])
            log["eval_nob_exact"].append(rn["exact_match"])
            log["eval_b_leak"].append(rb["b_leak_rate"])
            # RL curves swing, so keep the best checkpoint seen at eval time
            score = rb["exact_match"] + rn["exact_match"]
            if score > best_score:
                best_score = score
                save_model(policy, os.path.join(CKPT_DIR, f"grpo_{args.tag or args.reward}_best.pt"),
                           extra={**vars(args), "best_step": step})
            print(f"[eval] step {step:4d} | bomb-words exact {rb['exact_match']*100:6.2f}% "
                  f"| b-leak {rb['b_leak_rate']*100:6.2f}% | no-b words exact {rn['exact_match']*100:6.2f}% "
                  f"| {time.time()-t0:5.1f}s")
        if step == args.steps:
            break

        # ---- 1) sample a batch of prompts, each repeated G times ----
        words = [sampler.sample(must_have_b=(i < int(args.b_frac * args.prompts)))
                 for i in range(args.prompts)]
        temps, max_new = args.temp, None
        if args.explore_frac > 0:
            # Two kinds of episodes in one batch:
            #  * normal: whole answer sampled at --temp (on-policy, the real task)
            #  * explore: the environment starts at a hard state (a correct partial answer
            #    that already passed a 'b'); the policy makes ONE decision, sampled at the
            #    higher --explore_temp, and gets its token reward immediately.
            #    Easy states stay quiet (all samples right -> advantage 0) while the rare
            #    correct guesses at hard states get a clean, undiluted signal.
            n_exp = int(args.explore_frac * len(words))
            exp_words, norm_words = words[:n_exp], words[n_exp:]
            words = norm_words + exp_words
            pref = [""] * (len(norm_words) * G)
            for w in exp_words:
                t, rev = target_skip_b(w), w[::-1]
                if "b" in rev:
                    k0 = len(rev[:rev.index("b")])          # answer chars written before 1st b
                    k = sampler.rng.randint(k0, len(t))     # any state after the skip
                else:
                    k = sampler.rng.randint(0, len(t))
                pref += [t[:k]] * G
            rep = [w for w in words for _ in range(G)]
            nN = len(norm_words) * G
            seqN, plN = generate(policy, rep[:nN], greedy=False, temperature=args.temp, device=device)
            seqE, plE = generate(policy, rep[nN:], greedy=False, temperature=args.explore_temp,
                                 device=device, prefixes=pref[nN:], max_new=1)
            seq, plen = torch.cat([seqN, seqE]), plN + plE
            temps = torch.tensor([args.temp] * nN + [args.explore_temp] * (len(rep) - nN), device=device)
            max_new = torch.tensor([MAX_NEW_TOKENS] * nN + [1] * (len(rep) - nN), device=device)
            outs = [parse_completion(seq[i], plen[i]) for i in range(nN)] + [None] * (len(rep) - nN)
        else:
         rep = [w for w in words for _ in range(G)]
         # start-state resets: some episodes start part-way through a correct answer
         # (like "reverse curriculum"), so the hard states right after a skipped 'b' are
         # visited often. Only tokens the policy generates itself are trained on.
         pref = None
         if args.reset_frac > 0:
             pref = []
             for w in words:
                 t = target_skip_b(w)
                 k = 0
                 if sampler.rng.random() < args.reset_frac:
                     # candidate start points: right after a skipped 'b' (the hard states),
                     # else any point inside the answer
                     rev = w[::-1]
                     hard = sorted({len(rev[:i].replace("b", "")) for i in range(1, len(rev))
                                    if rev[i - 1] == "b"} - {0})
                     hard = [h for h in hard if h < len(t)] or list(range(1, len(t)))
                     if hard:
                         k = sampler.rng.choice(hard)
                 pref += [t[:k]] * G
         seq, plen = generate(policy, rep, greedy=False, temperature=args.temp, device=device,
                              prefixes=pref)
         outs = [parse_completion(seq[i], plen[i]) for i in range(len(rep))]
         if pref:
             outs = [None if o is None else p + o for p, o in zip(pref, outs)]

        # ---- 2) rewards and group-relative advantages ----
        R = torch.tensor([reward_fn(w, o) for w, o in zip(rep, outs)], dtype=torch.float32)
        Rg = R.view(args.prompts, G)
        adv = ((Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)).view(-1)
        adv = adv.to(device)

        with torch.no_grad():
            old_logp, mask = completion_logprobs(policy, seq, plen, temps, max_new=max_new)
            ref_logp, _ = completion_logprobs(ref, seq, plen, temps, max_new=max_new)
            if args.adv == "token":
                tok_adv, _ = token_advantages(seq, plen, rep, G, mask, args.gamma, args.baseline, args.neg_scale)

        # ---- 3) clipped policy-gradient update (PPO-style) + KL to reference ----
        policy.train()
        for _ in range(args.inner):
            logp, _, ent = completion_logprobs(policy, seq, plen, temps, return_entropy=True,
                                               max_new=max_new)
            ratio = torch.exp(logp - old_logp)
            a = tok_adv if args.adv == "token" else adv.unsqueeze(1)
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
