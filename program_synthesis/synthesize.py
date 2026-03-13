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
    generate_random_program, generate_random_safe_program,
    step, run, encode, decode, value_to_bytes, bytes_to_value,
)
from subleq.interpreter import MEM_SIZE, VOCAB_SIZE, SEQ_LEN, BYTES_PER_VALUE, CODE_SIZE

HALT_TOKEN = (-1) & 0xFF  # 255
ADDR_TOKENS = list(range(MEM_SIZE))  # 0..31
BRANCH_TOKENS = ADDR_TOKENS + [HALT_TOKEN]  # 0..31 plus -1


def make_valid_masks(n_code, device, offset=0):
    """Per-cell masks: a/b operands allow 0..31, c operands also allow -1."""
    ab_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    ab_mask[ADDR_TOKENS] = True
    c_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    c_mask[BRANCH_TOKENS] = True
    masks = []
    for i in range(n_code):
        masks.append(c_mask if (offset + i) % 3 == 2 else ab_mask)
    return masks


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


def hybrid_forward(model, fixed_tokens, code_params, code_positions, tau, mode="gumbel",
                   valid_masks=None):
    """Forward pass with learnable embeddings at code positions.

    valid_masks: list of per-position boolean masks (one per code cell),
                 or None for unconstrained.
    """
    B, S = fixed_tokens.shape

    tok_emb = model.token_emb(fixed_tokens).clone()

    for i, pos in enumerate(code_positions):
        if mode == "latent":
            tok_emb[:, pos] = code_params[i].unsqueeze(0).expand(B, -1)
        else:
            logits_i = code_params[i].unsqueeze(0).expand(B, -1)
            if valid_masks is not None:
                logits_i = logits_i.masked_fill(~valid_masks[i].unsqueeze(0), float('-inf'))
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


def _masked_argmax(code_params, valid_masks):
    """Argmax over logits with optional per-position masking."""
    if valid_masks is None:
        return code_params.argmax(dim=-1).tolist()
    result = []
    for i in range(code_params.shape[0]):
        logits_i = code_params[i].masked_fill(~valid_masks[i], float('-inf'))
        result.append(logits_i.argmax().item())
    return result


def decode_latent(code_params, emb_weight, valid_masks=None):
    """Map latent vectors back to token ids via nearest embedding."""
    sims = F.cosine_similarity(
        code_params.unsqueeze(1), emb_weight.unsqueeze(0), dim=-1)
    if valid_masks is not None:
        for i in range(sims.shape[0]):
            sims[i] = sims[i].masked_fill(~valid_masks[i], float('-inf'))
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
    elif program in ("random", "random_safe"):
        old_state = random.getstate()
        random.seed(rng.randint(0, 2**32 - 1))
        if program == "random_safe":
            mem, pc = generate_random_safe_program(value_range=30)
        else:
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


def symbolic_acc(pred_code, pairs, code_offset, n_code_opt):
    """Run decoded program on real SUBLEQ interpreter for each I/O pair."""
    correct = 0
    data_start = code_offset + n_code_opt
    for _, _, mem_i, pc_i, new_mem_i, new_pc_i in pairs:
        test_mem = list(mem_i)
        test_mem[code_offset:code_offset + n_code_opt] = pred_code
        try:
            out_m, out_pc, halted = step(test_mem, pc_i)
        except Exception:
            continue
        if out_pc == new_pc_i and out_m[data_start:] == new_mem_i[data_start:]:
            correct += 1
    return correct, len(pairs)


def symbolic_acc_indices(pred_code, pairs, code_offset, n_code_opt, indices):
    """Symbolic accuracy on a subset of pairs given by indices."""
    correct = 0
    data_start = code_offset + n_code_opt
    for i in indices:
        _, _, mem_i, pc_i, new_mem_i, new_pc_i = pairs[i]
        test_mem = list(mem_i)
        test_mem[code_offset:code_offset + n_code_opt] = pred_code
        try:
            out_m, out_pc, halted = step(test_mem, pc_i)
        except Exception:
            continue
        if out_pc == new_pc_i and out_m[data_start:] == new_mem_i[data_start:]:
            correct += 1
    return correct, len(indices)


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
    parser.add_argument("--code-offset", type=int, default=0,
                        help="Starting code cell index to optimize (default 0)")
    parser.add_argument("--n-io", type=int, default=1, help="Number of I/O pairs (sampled from execution traces)")
    parser.add_argument("--n-test", type=int, default=0, help="Number of held-out test I/O pairs (0 to skip)")
    parser.add_argument("--program",
                        choices=["negate", "addition", "countdown", "multiply",
                                 "fibonacci", "div", "isqrt", "chain", "random", "random_safe"],
                        default="negate")
    parser.add_argument("--constrained", action="store_true",
                        help="Restrict code tokens to valid SUBLEQ addresses (0-31 and -1)")
    parser.add_argument("--pc", type=int, nargs="+", default=None,
                        help="Only train on I/O pairs at these PC values (e.g. --pc 12 or --pc 0 12)")
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
    n_code_total = CODE_SIZE if args.code_cells == "all" else int(args.code_cells)
    code_offset = args.code_offset
    n_code_opt = n_code_total  # number of cells to optimize
    # code_positions: token indices of the cells we optimize (pos 0 = PC, pos 1 = mem[0], ...)
    code_positions = list(range(1 + code_offset, 1 + code_offset + n_code_opt))
    # n_code: total code region for trace filtering (must cover offset + count)
    n_code = code_offset + n_code_opt

    # --- Create I/O pairs from execution traces ---
    pc_filter = set(args.pc) if args.pc is not None else None
    rng = random.Random(args.seed)
    io_pairs = []
    first_mem = None
    while len(io_pairs) < args.n_io:
        if first_mem is None:
            # For random programs, retry until we get one that runs >= 3 steps
            for _ in range(100):
                mem, pc = make_program_state(args.program, rng)
                test_trace = make_trace_pairs(mem, pc)
                if args.program not in ("random", "random_safe") or len(test_trace) >= 3:
                    break
            first_mem = mem
            start_pc = pc
        elif args.program in ("random", "random_safe"):
            mem = list(first_mem)
            for j in range(CODE_SIZE, MEM_SIZE):
                mem[j] = rng.randint(-30, 30)
            pc = start_pc
        else:
            mem, pc = make_program_state(args.program, rng)
        trace = make_trace_pairs(mem, pc)
        # Only keep steps where the executed instruction is in the optimized region
        trace = [p for p in trace if p[3] < n_code]
        if args.pc is not None:
            trace = [p for p in trace if p[3] in pc_filter]
        io_pairs.extend(trace)
    io_pairs = io_pairs[:args.n_io]
    true_code = first_mem[code_offset:code_offset + n_code_opt]

    print(f"\nProgram: {args.program}", end="")
    if args.pc is not None:
        print(f"  (filtered to pc={sorted(pc_filter)})", end="")
    print()
    print(f"Ground truth code cells mem[{code_offset}..{code_offset + n_code_opt - 1}]:")
    for i in range(0, n_code_opt, 3):
        chunk = true_code[i:i+3]
        abs_i = code_offset + i
        print(f"  instr @{abs_i}: ({', '.join(str(v) for v in chunk)})")

    # Show which PCs are covered
    pcs_covered = sorted(set(p[3] for p in io_pairs))
    print(f"\n{len(io_pairs)} I/O pair(s) covering pc={pcs_covered}:")
    for inp, tgt, mem, pc, new_mem, new_pc in io_pairs[:10]:
        a, b = mem[pc], mem[pc + 1]
        print(f"  pc {pc}->{new_pc}: mem[{b}] {mem[b]}->{new_mem[b]}")
    if len(io_pairs) > 10:
        print(f"  ... and {len(io_pairs) - 10} more")

    # Per-instruction signal analysis
    data_cell_indices = set(range(CODE_SIZE, MEM_SIZE))
    for pc_val in sorted(set(p[3] for p in io_pairs)):
        pc_pairs = [p for p in io_pairs if p[3] == pc_val]
        n_changes = sum(
            1 for _, _, mem_i, _, new_mem_i, new_pc_i in pc_pairs
            if any(mem_i[j] != new_mem_i[j] for j in data_cell_indices)
        )
        print(f"  @{pc_val}: {len(pc_pairs)} examples, {n_changes} change data ({n_changes*100//max(len(pc_pairs),1)}%)")

    # Group I/O pair indices by PC for per-instruction reporting
    pc_values = sorted(set(p[3] for p in io_pairs))
    train_pc_indices = {pc: [i for i, p in enumerate(io_pairs) if p[3] == pc] for pc in pc_values}

    inp_batch = torch.stack([p[0] for p in io_pairs]).to(device)
    tgt_batch = torch.stack([p[1] for p in io_pairs]).to(device)

    # --- Generate held-out test I/O pairs ---
    test_pairs = []
    test_inp_batch = test_tgt_batch = None
    if args.n_test > 0:
        test_rng = random.Random(args.seed + 9999)
        while len(test_pairs) < args.n_test:
            if args.program in ("random", "random_safe"):
                mem = list(first_mem)
                for j in range(CODE_SIZE, MEM_SIZE):
                    mem[j] = test_rng.randint(-30, 30)
                pc = start_pc
            else:
                mem, pc = make_program_state(args.program, test_rng)
            trace = make_trace_pairs(mem, pc)
            filtered = [p for p in trace if p[3] < n_code]
            if pc_filter is not None:
                filtered = [p for p in filtered if p[3] in pc_filter]
            test_pairs.extend(filtered)
        test_pairs = test_pairs[:args.n_test]
        test_inp_batch = torch.stack([p[0] for p in test_pairs]).to(device)
        test_tgt_batch = torch.stack([p[1] for p in test_pairs]).to(device)
        test_pc_indices = {pc: [i for i, p in enumerate(test_pairs) if p[3] == pc] for pc in pc_values}
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

    # --- Constraint masks ---
    valid_masks = make_valid_masks(n_code_opt, device, code_offset) if args.constrained else None
    if args.constrained:
        print(f"\nConstrained: a,b in [0,{MEM_SIZE-1}] ({MEM_SIZE} tokens), c also allows -1 ({MEM_SIZE+1} tokens)")

    # --- Set up learnable parameters ---
    if args.mode == "latent":
        code_params = nn.Parameter(torch.randn(n_code_opt, model.d_model, device=device) * 0.02)
    else:
        code_params = nn.Parameter(torch.zeros(n_code_opt, VOCAB_SIZE, device=device))
    optimizer = torch.optim.Adam([code_params], lr=args.lr)

    constrained_tag = "+constrained" if args.constrained else ""
    offset_tag = f" (cells {code_offset}..{code_offset + n_code_opt - 1})" if code_offset > 0 else ""
    print(f"\nOptimizing {n_code_opt} code cells{offset_tag} over {args.steps} steps")
    print(f"  mode={args.mode}{constrained_tag}", end="")
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
            model, inp_batch, code_params, code_positions, tau, mode=args.mode,
            valid_masks=valid_masks)
        loss = F.cross_entropy(
            out_logits[:, data_positions].reshape(-1, VOCAB_SIZE),
            tgt_batch[:, data_positions].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_i % 100 == 0 or step_i == 1:
            with torch.no_grad():
                if args.mode == "latent":
                    guesses = decode_latent(code_params, emb_weight, valid_masks)
                else:
                    guesses = _masked_argmax(code_params, valid_masks)
                decoded = [bytes_to_value([g]) for g in guesses]
                n_match = sum(a == b for a, b in zip(decoded, true_code))  # true_code is the optimized slice

                tau_str = f" | tau={tau:.3f}" if args.mode != "latent" else ""

                # Neural executor accuracy (discrete tokens through model)
                tok_ids = guesses if args.mode != "latent" else decode_latent(code_params, emb_weight, valid_masks)
                def neural_acc(inp, tgt):
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

                n_tr = neural_acc(inp_batch, tgt_batch)
                s_tr_c, s_tr_t = symbolic_acc(decoded, io_pairs, code_offset, n_code_opt)
                s_tr = s_tr_c / s_tr_t
                acc_str = f"neural={n_tr:.0%} sym={s_tr:.0%}"
                if test_inp_batch is not None:
                    n_te = neural_acc(test_inp_batch, test_tgt_batch)
                    s_te_c, s_te_t = symbolic_acc(decoded, test_pairs, code_offset, n_code_opt)
                    s_te = s_te_c / s_te_t
                    acc_str += f" | test: neural={n_te:.0%} sym={s_te:.0%}"
                print(f"step {step_i:5d} | loss={loss.item():.4f}{tau_str} | "
                      f"{n_match}/{n_code_opt} cells | {acc_str}")
                # Per-instruction breakdown
                parts = []
                for pc_val in pc_values:
                    tr_c, tr_t = symbolic_acc_indices(decoded, io_pairs, code_offset, n_code_opt, train_pc_indices[pc_val])
                    part = f"@{pc_val}:{tr_c}/{tr_t}"
                    if test_inp_batch is not None and pc_val in test_pc_indices and test_pc_indices[pc_val]:
                        te_c, te_t = symbolic_acc_indices(decoded, test_pairs, code_offset, n_code_opt, test_pc_indices[pc_val])
                        part += f"({te_c}/{te_t})"
                    parts.append(part)
                print(f"  {'':>9} per-instr sym: {' '.join(parts)}")

    # --- Final result ---
    with torch.no_grad():
        if args.mode == "latent":
            final_tokens = decode_latent(code_params, emb_weight, valid_masks)
        else:
            final_tokens = _masked_argmax(code_params, valid_masks)
    pred_code = [bytes_to_value([t]) for t in final_tokens]

    has_test = test_inp_batch is not None
    hdr = f"{'Instr':>10} | {'Recovered':>20} | {'Target':>20} | Match | {'Train sym':>10}"
    if has_test:
        hdr += f" | {'Test sym':>10}"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'-'*len(hdr)}")
    for i in range(0, n_code_opt, 3):
        pred_chunk = pred_code[i:i+3]
        true_chunk = true_code[i:i+3]
        match = pred_chunk == true_chunk
        abs_i = code_offset + i
        if abs_i in train_pc_indices and train_pc_indices[abs_i]:
            tr_c, tr_t = symbolic_acc_indices(pred_code, io_pairs, code_offset, n_code_opt, train_pc_indices[abs_i])
            tr_str = f"{tr_c}/{tr_t}"
        else:
            tr_str = "n/a"
        row = (f"{'@'+str(abs_i):>10} | {str(tuple(pred_chunk)):>20} | {str(tuple(true_chunk)):>20}"
               f" | {'yes' if match else 'NO':>5} | {tr_str:>10}")
        if has_test:
            if abs_i in test_pc_indices and test_pc_indices[abs_i]:
                te_c, te_t = symbolic_acc_indices(pred_code, test_pairs, code_offset, n_code_opt, test_pc_indices[abs_i])
                te_str = f"{te_c}/{te_t}"
            else:
                te_str = "n/a"
            row += f" | {te_str:>10}"
        print(row)

    n_diff = sum(a != b for a, b in zip(pred_code, true_code))
    if n_diff == 0:
        print(f"\nEXACT MATCH — all {n_code_opt} cells recovered")
    else:
        print(f"\n{n_diff}/{n_code_opt} cells differ from ground truth")

    with torch.no_grad():
        tok_ids = final_tokens
        def final_neural_acc(inp, tgt):
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
            return correct, inp.shape[0]

        n_tr_c, n_tr_t = final_neural_acc(inp_batch, tgt_batch)
        s_tr_c, s_tr_t = symbolic_acc(pred_code, io_pairs, code_offset, n_code_opt)
        print(f"\nTrain  neural: {n_tr_c}/{n_tr_t} ({n_tr_c/n_tr_t:.0%})  symbolic: {s_tr_c}/{s_tr_t} ({s_tr_c/s_tr_t:.0%})")
        if has_test:
            n_te_c, n_te_t = final_neural_acc(test_inp_batch, test_tgt_batch)
            s_te_c, s_te_t = symbolic_acc(pred_code, test_pairs, code_offset, n_code_opt)
            print(f"Test   neural: {n_te_c}/{n_te_t} ({n_te_c/n_te_t:.0%})  symbolic: {s_te_c}/{s_te_t} ({s_te_c/s_te_t:.0%})")


if __name__ == "__main__":
    main()
