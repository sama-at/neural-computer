#!/usr/bin/env python3
"""
Program synthesis via backpropagation through a learned SUBLEQ executor.

Level 1: Recover one instruction (a, b, c) from a single I/O pair.
Uses soft token embeddings (Gumbel-softmax or plain softmax with temperature)
to make discrete tokens differentiable.
"""

import sys
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'round2_trained'))

from subleq import (
    MiniSUBLEQTransformer,
    make_negate, make_addition, make_countdown, make_multiply,
    make_fibonacci, make_div, make_isqrt, make_chain,
    generate_random_program,
    step, run, encode, decode, value_to_bytes, bytes_to_value,
)
from subleq.interpreter import MEM_SIZE, VOCAB_SIZE, SEQ_LEN, BYTES_PER_VALUE, CODE_SIZE


def load_executor(checkpoint_path, device):
    """Load and freeze the trained executor."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    model = MiniSUBLEQTransformer(**config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def random_executor(device):
    """Create a randomly initialized (untrained) executor as a control."""
    config = dict(d_model=256, n_heads=8, n_layers=6, d_ff=1024)
    model = MiniSUBLEQTransformer(**config).to(device)
    print(f"Random executor (untrained): {model.count_params():,} params")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def hybrid_forward(model, fixed_tokens, code_params, code_positions, tau, mode="gumbel"):
    """Forward pass with learnable embeddings at code positions.

    Modes:
      gumbel:       soft Gumbel-softmax (continuous blend of embeddings)
      gumbel_hard:  hard Gumbel-softmax (discrete forward, soft backward)
      softmax:      plain softmax with temperature
      latent:       optimize directly in embedding space
    """
    B, S = fixed_tokens.shape

    tok_emb = model.token_emb(fixed_tokens).clone()

    for i, pos in enumerate(code_positions):
        if mode == "latent":
            tok_emb[:, pos] = code_params[i].unsqueeze(0).expand(B, -1)
        else:
            logits_i = code_params[i].unsqueeze(0).expand(B, -1)
            if mode in ("gumbel", "gumbel_hard"):
                hard = mode == "gumbel_hard"
                probs = F.gumbel_softmax(logits_i, tau=tau, hard=hard)
            else:
                probs = F.softmax(logits_i / tau, dim=-1)
            tok_emb[:, pos] = probs @ model.token_emb.weight

    pos_emb = model.pos_emb(model.pos_indices[:, :S].expand(B, -1))
    typ_emb = model.type_emb(model.type_indices[:, :S].expand(B, -1))
    h = tok_emb + pos_emb + typ_emb

    for layer in model.layers:
        h = layer(h)

    h = model.final_norm(h)
    return model.output_head(h)


def decode_latent(code_params, emb_weight):
    """Map latent vectors back to token ids via nearest embedding."""
    # code_params: (N, d_model), emb_weight: (vocab, d_model)
    sims = F.cosine_similarity(
        code_params.unsqueeze(1), emb_weight.unsqueeze(0), dim=-1)
    return sims.argmax(dim=-1).tolist()


def make_program_state(program, rng):
    """Create initial (mem, pc) for a program with random data values."""
    if program == "negate":
        val = rng.randint(-100, 100)
        result = make_negate(val)
    elif program == "addition":
        a, b = rng.randint(-60, 60), rng.randint(-60, 60)
        result = make_addition(a, b)
    elif program == "countdown":
        start = rng.randint(1, 20)
        result = make_countdown(start)
    elif program == "multiply":
        a = rng.randint(1, 11)
        b = rng.randint(1, 127 // max(a, 1))
        result = make_multiply(a, b)
    elif program == "fibonacci":
        n = rng.randint(1, 10)
        result = make_fibonacci(n)
    elif program == "div":
        b = rng.randint(1, 20)
        a = rng.randint(b, 126)
        result = make_div(a, b)
    elif program == "isqrt":
        n = rng.randint(1, 126)
        result = make_isqrt(n)
    elif program == "chain":
        vals = [rng.choice([-1, 1]) * rng.randint(1, 30) for _ in range(8)]
        result = make_chain(values=vals)
    elif program == "random":
        old_state = random.getstate()
        random.seed(rng.randint(0, 2**32 - 1))
        mem, pc = generate_random_program(value_range=30)
        random.setstate(old_state)
        return mem, pc
    else:
        raise ValueError(f"Unknown program: {program}")
    return result[0], result[1]


def make_trace_pairs(mem, pc, max_steps=500):
    """Generate one I/O pair per execution step until halt or step limit."""
    pairs = []
    m = list(mem)
    for _ in range(max_steps):
        inp = encode(m, pc)
        new_m, new_pc, halted = step(m, pc)
        tgt = encode(new_m, new_pc)
        pairs.append((inp, tgt, list(m), pc, list(new_m), new_pc))
        if halted:
            break
        m, pc = new_m, new_pc
    return pairs


def check_equivalence(pred_code, true_code, io_pairs, n_code):
    """Return 'exact', 'equivalent', 'wrong', or 'error'."""
    if pred_code == true_code:
        return "exact"
    seen = set()
    for _, _, mem_i, pc_i, _, _ in io_pairs:
        if pc_i != 0:
            continue
        key = tuple(mem_i[n_code:])
        if key in seen:
            continue
        seen.add(key)
        gt_final, gt_pc, _ = run(mem_i, 0)
        test_mem = list(mem_i)
        test_mem[:n_code] = pred_code
        pred_final, pred_pc, pred_steps = run(test_mem, 0)
        if pred_steps == 0:
            return "error"
        if pred_final[n_code:] != gt_final[n_code:] or pred_pc != gt_pc:
            return "wrong"
    return "equivalent"


def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize a SUBLEQ instruction by backprop through a learned executor")
    parser.add_argument("--checkpoint", default="round2_trained/checkpoints/best_model.pt",
                        help="Path to trained model checkpoint, or 'none' to train from scratch")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-start", type=float, default=5.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--mode", choices=["gumbel", "gumbel_hard", "softmax", "latent"],
                        default="gumbel",
                        help="gumbel: soft Gumbel-softmax; "
                             "gumbel_hard: hard Gumbel (discrete fwd, soft bwd); "
                             "softmax: plain softmax with temperature; "
                             "latent: optimize directly in embedding space")
    parser.add_argument("--code-cells", type=str, default="3",
                        help="Number of code cells to optimize: integer or 'all' for full code region")
    parser.add_argument("--n-io", type=int, default=1, help="Number of I/O pairs (sampled from execution traces)")
    parser.add_argument("--n-test", type=int, default=0, help="Number of held-out test I/O pairs (0 to skip)")
    parser.add_argument("--program",
                        choices=["negate", "addition", "countdown", "multiply",
                                 "fibonacci", "div", "isqrt", "chain", "random"],
                        default="negate")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        args.device = auto_device()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # --- Load or train frozen executor ---
    if args.checkpoint.lower() == "none":
        model = random_executor(device)
    else:
        print(f"Loading executor from {args.checkpoint}")
        model = load_executor(args.checkpoint, device)
    print(f"Model: {model.count_params():,} params (frozen)")

    # --- Parse code cells ---
    n_code = CODE_SIZE if args.code_cells == "all" else int(args.code_cells)
    code_positions = list(range(1, 1 + n_code))  # token indices (pos 0 = PC)

    # --- Create I/O pairs from execution traces ---
    rng = random.Random(args.seed)
    io_pairs = []
    first_mem = None
    while len(io_pairs) < args.n_io:
        if first_mem is None:
            # For random programs, retry until we get one that runs >= 3 steps
            for _ in range(100):
                mem, pc = make_program_state(args.program, rng)
                test_trace = make_trace_pairs(mem, pc)
                if args.program != "random" or len(test_trace) >= 3:
                    break
            first_mem = mem
            start_pc = pc
        elif args.program == "random":
            # Keep code, re-randomize data cells
            mem = list(first_mem)
            for j in range(CODE_SIZE, MEM_SIZE):
                mem[j] = rng.randint(-30, 30)
            pc = start_pc
        else:
            mem, pc = make_program_state(args.program, rng)
        trace = make_trace_pairs(mem, pc)
        # Only keep steps where the executed instruction is in the optimized region
        trace = [p for p in trace if p[3] < n_code]
        io_pairs.extend(trace)
    io_pairs = io_pairs[:args.n_io]
    true_code = first_mem[:n_code]

    print(f"\nProgram: {args.program}")
    print(f"Ground truth code cells mem[0..{n_code-1}]:")
    for i in range(0, n_code, 3):
        chunk = true_code[i:i+3]
        print(f"  instr @{i}: ({', '.join(str(v) for v in chunk)})")

    # Show which PCs are covered
    pcs_covered = sorted(set(p[3] for p in io_pairs))
    print(f"\n{len(io_pairs)} I/O pair(s) covering pc={pcs_covered}:")
    for inp, tgt, mem, pc, new_mem, new_pc in io_pairs[:10]:
        a, b = mem[pc], mem[pc + 1]
        print(f"  pc {pc}->{new_pc}: mem[{b}] {mem[b]}->{new_mem[b]}")
    if len(io_pairs) > 10:
        print(f"  ... and {len(io_pairs) - 10} more")

    inp_batch = torch.stack([p[0] for p in io_pairs]).to(device)
    tgt_batch = torch.stack([p[1] for p in io_pairs]).to(device)

    # --- Generate held-out test I/O pairs ---
    test_inp_batch = test_tgt_batch = None
    if args.n_test > 0:
        test_rng = random.Random(args.seed + 9999)
        test_pairs = []
        while len(test_pairs) < args.n_test:
            if args.program == "random":
                mem = list(first_mem)
                for j in range(CODE_SIZE, MEM_SIZE):
                    mem[j] = test_rng.randint(-30, 30)
                pc = start_pc
            else:
                mem, pc = make_program_state(args.program, test_rng)
            trace = make_trace_pairs(mem, pc)
            test_pairs.extend([p for p in trace if p[3] < n_code])
        test_pairs = test_pairs[:args.n_test]
        test_inp_batch = torch.stack([p[0] for p in test_pairs]).to(device)
        test_tgt_batch = torch.stack([p[1] for p in test_pairs]).to(device)
        print(f"{args.n_test} held-out test I/O pairs")

    # --- Verify executor works on the correct inputs ---
    with torch.no_grad():
        logits = model(inp_batch)
        preds = logits.argmax(dim=-1)
        per_example = (preds == tgt_batch).all(dim=1)
        n_perfect = per_example.sum().item()
        print(f"\nExecutor sanity check: {n_perfect}/{args.n_io} examples perfect", end="")
        if n_perfect == args.n_io:
            print()
        else:
            print(" WARNING — model may be undertrained")

    # --- Set up learnable parameters ---
    if args.mode == "latent":
        code_params = nn.Parameter(torch.randn(n_code, model.d_model, device=device) * 0.02)
    else:
        code_params = nn.Parameter(torch.zeros(n_code, VOCAB_SIZE, device=device))
    optimizer = torch.optim.Adam([code_params], lr=args.lr)

    print(f"\nOptimizing {n_code} code cells over {args.steps} steps")
    print(f"  mode={args.mode}", end="")
    if args.mode != "latent":
        print(f", tau: {args.tau_start} -> {args.tau_end}")
    else:
        print(f", d_model={model.d_model}")
    print()

    emb_weight = model.token_emb.weight

    # Loss only on observable behavior: PC (pos 0) + data cells (pos 25-32)
    data_positions = [0] + list(range(1 + CODE_SIZE, SEQ_LEN))
    pos_labels = ["PC"] + [f"m{i}" for i in range(CODE_SIZE, MEM_SIZE)]
    print(f"  loss on {len(data_positions)} positions: {pos_labels}")

    # Show a few I/O examples at the labeled positions
    n_show = min(5, len(io_pairs))
    print(f"\n  Example I/O at supervised positions (first {n_show}):")
    print(f"  {'':>6}  {' '.join(f'{l:>5}' for l in pos_labels)}")
    for ex_i in range(n_show):
        inp_vals = [bytes_to_value([inp_batch[ex_i, p].item()]) for p in data_positions]
        tgt_vals = [bytes_to_value([tgt_batch[ex_i, p].item()]) for p in data_positions]
        print(f"  in {ex_i}: {' '.join(f'{v:>5}' for v in inp_vals)}")
        print(f"  out{ex_i}: {' '.join(f'{v:>5}' for v in tgt_vals)}")
        changed = [l for l, iv, tv in zip(pos_labels, inp_vals, tgt_vals) if iv != tv]
        print(f"  {'':>6}  changed: {changed if changed else '(none)'}")
    print()

    # --- Optimization loop ---
    for step_i in range(1, args.steps + 1):
        frac = step_i / args.steps
        tau = args.tau_start * (args.tau_end / args.tau_start) ** frac

        out_logits = hybrid_forward(
            model, inp_batch, code_params, code_positions, tau, mode=args.mode)
        loss = F.cross_entropy(
            out_logits[:, data_positions].reshape(-1, VOCAB_SIZE),
            tgt_batch[:, data_positions].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_i % 100 == 0 or step_i == 1:
            with torch.no_grad():
                if args.mode == "latent":
                    guesses = decode_latent(code_params, emb_weight)
                else:
                    guesses = code_params.argmax(dim=-1).tolist()
                decoded = [bytes_to_value([g]) for g in guesses]
                n_match = sum(a == b for a, b in zip(decoded, true_code))
                status = check_equivalence(decoded, true_code, io_pairs, n_code)

                if args.mode == "latent":
                    sims = F.cosine_similarity(
                        code_params.unsqueeze(1), emb_weight.unsqueeze(0), dim=-1)
                    avg_sim = sims.max(dim=-1).values.mean().item()
                    conf_str = f"sim={avg_sim:.2f}"
                else:
                    probs = F.softmax(code_params, dim=-1)
                    avg_conf = probs.max(dim=-1).values.mean().item()
                    conf_str = f"conf={avg_conf:.2f}"

                tau_str = f" | tau={tau:.3f}" if args.mode != "latent" else ""
                status_str = {"exact": "EXACT", "equivalent": "EQUIV",
                              "wrong": "WRONG", "error": "ERROR"}[status]
                # Discrete eval: get token ids, build exact embeddings
                if args.mode == "latent":
                    tok_ids = decode_latent(code_params, emb_weight)
                else:
                    tok_ids = guesses  # already argmax'd above
                def discrete_acc(inp, tgt):
                    te = model.token_emb(inp).clone()
                    for ii, pos in enumerate(code_positions):
                        te[:, pos] = emb_weight[tok_ids[ii]]
                    B_t, S_t = inp.shape
                    h_t = te + model.pos_emb(model.pos_indices[:, :S_t].expand(B_t, -1)) + model.type_emb(model.type_indices[:, :S_t].expand(B_t, -1))
                    for layer in model.layers:
                        h_t = layer(h_t)
                    logits = model.output_head(model.final_norm(h_t))
                    preds = logits[:, data_positions].argmax(dim=-1)
                    tgts = tgt[:, data_positions]
                    return (preds == tgts).all(dim=1).float().mean().item()

                train_acc = discrete_acc(inp_batch, tgt_batch)
                train_str = f" | train={train_acc:.0%}"
                test_str = ""
                if test_inp_batch is not None:
                    test_acc = discrete_acc(test_inp_batch, test_tgt_batch)
                    test_str = f" | test={test_acc:.0%}"
                print(f"step {step_i:5d} | loss={loss.item():.4f}{tau_str} | "
                      f"{n_match}/{n_code} cells | {conf_str} {status_str}{train_str}{test_str}")

    # --- Final result ---
    with torch.no_grad():
        if args.mode == "latent":
            final_tokens = decode_latent(code_params, emb_weight)
        else:
            final_tokens = code_params.argmax(dim=-1).tolist()
    pred_code = [bytes_to_value([t]) for t in final_tokens]

    print(f"\n{'='*60}")
    print(f"{'Instr':>10} | {'Recovered':>20} | {'Target':>20} | Match")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*20}-+------")
    all_match = True
    for i in range(0, n_code, 3):
        pred_chunk = pred_code[i:i+3]
        true_chunk = true_code[i:i+3]
        match = pred_chunk == true_chunk
        if not match:
            all_match = False
        print(f"{'@'+str(i):>10} | {str(tuple(pred_chunk)):>20} | {str(tuple(true_chunk)):>20} | {'yes' if match else 'NO'}")

    final_status = check_equivalence(pred_code, true_code, io_pairs, n_code)
    n_diff = sum(a != b for a, b in zip(pred_code, true_code))
    if final_status == "exact":
        print("\nEXACT MATCH")
    elif final_status == "equivalent":
        print(f"\nEQUIVALENT — {n_diff} code cell(s) differ but full execution matches on all inputs")
    elif final_status == "error":
        print(f"\nERROR — recovered program crashes (invalid addresses)")
    else:
        print(f"\nWRONG — {n_diff}/{n_code} cells wrong")

    with torch.no_grad():
        if args.mode == "latent":
            tok_ids = decode_latent(code_params, emb_weight)
        else:
            tok_ids = code_params.argmax(dim=-1).tolist()
        def final_acc(inp, tgt):
            te = model.token_emb(inp).clone()
            for ii, pos in enumerate(code_positions):
                te[:, pos] = emb_weight[tok_ids[ii]]
            B_t, S_t = inp.shape
            h_t = te + model.pos_emb(model.pos_indices[:, :S_t].expand(B_t, -1)) + model.type_emb(model.type_indices[:, :S_t].expand(B_t, -1))
            for layer in model.layers:
                h_t = layer(h_t)
            logits = model.output_head(model.final_norm(h_t))
            preds = logits[:, data_positions].argmax(dim=-1)
            tgts = tgt[:, data_positions]
            correct = (preds == tgts).all(dim=1).sum().item()
            total = inp.shape[0]
            return correct, total
        train_c, train_t = final_acc(inp_batch, tgt_batch)
        print(f"\nTrain accuracy: {train_c}/{train_t} ({train_c/train_t:.0%})")
        if test_inp_batch is not None:
            test_c, test_t = final_acc(test_inp_batch, test_tgt_batch)
            print(f"Test accuracy:  {test_c}/{test_t} ({test_c/test_t:.0%})")


if __name__ == "__main__":
    main()
