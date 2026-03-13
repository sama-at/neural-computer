#!/usr/bin/env python3
"""Multi-step program synthesis via backpropagation through a chained differentiable executor.

Instead of single-step I/O, chains the neural executor for k steps and
backpropagates loss from the final state through all intermediate steps.
This recovers branch targets and program features that are underdetermined
from single-step observations alone.

Usage:
    # Multi-step (8 chained steps):
    python synthesize_multistep.py --program fibonacci --exec-steps 8 --constrained
    # Single-step baseline (for comparison):
    python synthesize_multistep.py --program fibonacci --exec-steps 1 --constrained
"""

import sys
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'round2_trained'))

from subleq import step, encode, value_to_bytes, bytes_to_value
from subleq.interpreter import MEM_SIZE, VOCAB_SIZE, SEQ_LEN, CODE_SIZE, DATA_START
from subleq.programs import make_chain

sys.path.insert(0, os.path.dirname(__file__))
from synthesize import (
    load_executor, random_executor, make_valid_masks, make_program_state,
    _masked_argmax, decode_latent, auto_device,
)


DATA_ADDR_TOKENS = list(range(DATA_START, MEM_SIZE))  # 24..31
INSTR_ALIGNED_TOKENS = list(range(0, CODE_SIZE, 3)) + [(-1) & 0xFF]  # 0,3,6,...,21, 255


def make_data_only_masks(n_code, device, offset=0):
    """Tight masks for programs that only touch data memory.

    a,b: only data cell addresses (24..31).
    c: instruction-aligned jumps (0,3,6,...,21) plus halt (-1).
    """
    ab_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    for t in DATA_ADDR_TOKENS:
        ab_mask[t] = True
    c_mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    for t in INSTR_ALIGNED_TOKENS:
        c_mask[t] = True
    masks = []
    for i in range(n_code):
        masks.append(c_mask if (offset + i) % 3 == 2 else ab_mask)
    return masks


def forward_from_emb(model, tok_emb):
    """Forward pass from pre-computed token embeddings (bypasses model.token_emb)."""
    B, S, _ = tok_emb.shape
    pos = model.pos_emb(model.pos_indices[:, :S].expand(B, -1))
    typ = model.type_emb(model.type_indices[:, :S].expand(B, -1))
    h = tok_emb + pos + typ
    for layer in model.layers:
        h = layer(h)
    return model.output_head(model.final_norm(h))


def next_state_emb(model, logits, state_mode, state_tau):
    """Map logits to embeddings for the next unrolled step."""
    emb_w = model.token_emb.weight
    if state_mode == "soft":
        return F.softmax(logits / state_tau, dim=-1) @ emb_w
    if state_mode == "gumbel":
        probs = F.gumbel_softmax(logits, tau=state_tau, hard=False)
        return probs @ emb_w
    if state_mode == "gumbel_hard":
        probs = F.gumbel_softmax(logits, tau=state_tau, hard=True)
        return probs @ emb_w
    hard_emb = model.token_emb(logits.argmax(-1))
    if state_mode == "hard":
        return hard_emb
    soft_emb = F.softmax(logits / state_tau, dim=-1) @ emb_w
    return hard_emb + (soft_emb - soft_emb.detach())


DATA_TOKEN_POSITIONS = list(range(1 + CODE_SIZE, SEQ_LEN))  # positions 25..32


def chain_forward(model, initial_tokens, code_params, code_positions,
                  n_steps, tau, mode="gumbel", valid_masks=None,
                  state_mode="ste", state_tau=1.0, return_all=False,
                  intermediate_pcs=None, fixed_positions=None):
    """Chain the differentiable executor for n_steps.

    When intermediate_pcs is provided (no-branch mode), the intermediate state
    is constructed cleanly: only data cell embeddings come from the model's
    soft output; PC is hard-coded and all code positions are injected directly.
    """
    B, S = initial_tokens.shape
    emb_w = model.token_emb.weight

    code_embs = []
    for i in range(len(code_positions)):
        if mode == "latent":
            code_embs.append(code_params[i].unsqueeze(0).expand(B, -1))
        else:
            logits_i = code_params[i].unsqueeze(0).expand(B, -1)
            if valid_masks is not None:
                logits_i = logits_i.masked_fill(~valid_masks[i].unsqueeze(0), float('-inf'))
            if mode in ("gumbel", "gumbel_hard"):
                probs = F.gumbel_softmax(logits_i, tau=tau, hard=(mode == "gumbel_hard"))
            else:
                probs = F.softmax(logits_i / tau, dim=-1)
            code_embs.append(probs @ emb_w)

    orig_emb = model.token_emb(initial_tokens)
    tok_emb = orig_emb
    logits_steps = []

    for s in range(n_steps):
        e = tok_emb.clone()
        if intermediate_pcs is not None:
            # Per-step gradient: only the instruction at the current PC
            # gets gradients through the model; others are detached.
            instr_start = s * 3
            instr_end = min(instr_start + 3, len(code_positions))
            for i, pos in enumerate(code_positions):
                if instr_start <= i < instr_end:
                    e[:, pos] = code_embs[i]
                else:
                    e[:, pos] = code_embs[i].detach()
        else:
            for i, pos in enumerate(code_positions):
                e[:, pos] = code_embs[i]

        logits = forward_from_emb(model, e)
        logits_steps.append(logits)

        if s < n_steps - 1:
            if intermediate_pcs is not None:
                # No-branch: build intermediate state from scratch.
                # Only data cells come from the model's soft output.
                soft = next_state_emb(model, logits, state_mode, state_tau)
                tok_emb = orig_emb.clone()
                tok_emb[:, 0] = model.token_emb(intermediate_pcs[:, s + 1])
                tok_emb[:, DATA_TOKEN_POSITIONS] = soft[:, DATA_TOKEN_POSITIONS]
            else:
                tok_emb = next_state_emb(model, logits, state_mode, state_tau)

    if return_all:
        return logits_steps
    return logits_steps[-1]


def chain_discrete(model, inp, tok_ids, code_positions, n_steps,
                   return_all=False, intermediate_pcs=None,
                   fixed_positions=None):
    """Chain executor with discrete code tokens (for eval)."""
    B, S = inp.shape
    emb_w = model.token_emb.weight
    orig_emb = model.token_emb(inp)
    tok_emb = orig_emb
    logits_steps = []

    for s in range(n_steps):
        e = tok_emb.clone()
        for i, pos in enumerate(code_positions):
            e[:, pos] = emb_w[tok_ids[i]]
        logits = forward_from_emb(model, e)
        logits_steps.append(logits)
        if s < n_steps - 1:
            if intermediate_pcs is not None:
                pred_emb = model.token_emb(logits.argmax(-1))
                tok_emb = orig_emb.clone()
                tok_emb[:, 0] = model.token_emb(intermediate_pcs[:, s + 1])
                tok_emb[:, DATA_TOKEN_POSITIONS] = pred_emb[:, DATA_TOKEN_POSITIONS]
            else:
                tok_emb = model.token_emb(logits.argmax(-1))

    if return_all:
        return logits_steps
    return logits_steps[-1]


def make_trace_windows(mem, pc, k, max_steps=500, window_mode="any"):
    """Build contiguous k-step supervision windows from a full symbolic trace."""
    states = [(list(mem), pc)]
    m, p = list(mem), pc
    for _ in range(max_steps):
        m, p, halted = step(m, p)
        states.append((list(m), p))
        if halted:
            break

    n_exec = len(states) - 1
    if n_exec < k:
        return []

    if window_mode == "entry":
        starts = [0]
    else:
        starts = range(n_exec - k + 1)

    windows = []
    for start in starts:
        in_mem, in_pc = states[start]
        out_mem, out_pc = states[start + k]
        pcs = [states[start + j][1] for j in range(k + 1)]
        step_targets = [
            encode(states[start + j + 1][0], states[start + j + 1][1])
            for j in range(k)
        ]
        windows.append((
            encode(in_mem, in_pc),
            encode(out_mem, out_pc),
            list(in_mem),
            in_pc,
            list(out_mem),
            out_pc,
            pcs,
            torch.stack(step_targets),
        ))
    return windows


def filter_windows(windows, pc_filter=None, start_pc_filter=None, end_pc_filter=None):
    """Apply optional PC filters to a list of trace windows."""
    if pc_filter is not None:
        windows = [w for w in windows if any(p in pc_filter for p in w[6][:-1])]
    if start_pc_filter is not None:
        windows = [w for w in windows if w[3] in start_pc_filter]
    if end_pc_filter is not None:
        windows = [w for w in windows if w[6][-2] in end_pc_filter]
    return windows


def _window_bucket_key(window, sample_mode):
    if sample_mode == "path":
        return tuple(window[6])
    if sample_mode == "start_pc":
        return window[3]
    return None


def select_windows(windows, n_needed, rng, sample_mode):
    """Subsample candidate windows, optionally balancing by path statistics."""
    if len(windows) <= n_needed:
        out = list(windows)
        rng.shuffle(out)
        return out
    if sample_mode == "uniform":
        out = list(windows)
        rng.shuffle(out)
        return out[:n_needed]

    buckets = {}
    for window in windows:
        key = _window_bucket_key(window, sample_mode)
        buckets.setdefault(key, []).append(window)

    keys = list(buckets)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(buckets[key])

    selected = []
    while len(selected) < n_needed:
        made_progress = False
        for key in keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            made_progress = True
            if len(selected) == n_needed:
                break
        if not made_progress:
            break
        rng.shuffle(keys)
    return selected


def collect_windows(program, rng, n_needed, k, window_mode, sample_mode,
                    pc_filter=None, start_pc_filter=None, end_pc_filter=None,
                    first_mem=None, start_pc=None, overcollect=8,
                    chain_length=None):
    """Collect candidate windows from traces, then subsample them."""
    candidates = []
    base_mem = None if first_mem is None else list(first_mem)
    base_pc = start_pc
    attempts = 0
    target_candidates = n_needed if sample_mode == "uniform" else max(n_needed * overcollect, n_needed)

    def _make_state():
        if program == "chain" and chain_length is not None:
            vals = [rng.choice([-1, 1]) * rng.randint(1, 30) for _ in range(8)]
            mem, pc, _ = make_chain(num_instructions=chain_length, values=vals)
            return mem, pc
        return make_program_state(program, rng)

    while len(candidates) < target_candidates:
        if base_mem is None:
            mem, pc = _make_state()
            base_mem, base_pc = list(mem), pc
        elif program in ("random", "random_safe"):
            mem = list(base_mem)
            for j in range(CODE_SIZE, MEM_SIZE):
                mem[j] = rng.randint(-30, 30)
            pc = base_pc
        else:
            mem, pc = _make_state()

        windows = make_trace_windows(mem, pc, k, window_mode=window_mode)
        windows = filter_windows(
            windows,
            pc_filter=pc_filter,
            start_pc_filter=start_pc_filter,
            end_pc_filter=end_pc_filter,
        )
        if not windows:
            attempts += 1
            if attempts > max(n_needed * 20, 100):
                break
            continue
        candidates.extend(windows)
        attempts = 0

    selected = select_windows(candidates, n_needed, rng, sample_mode)
    return selected, base_mem, base_pc, len(candidates)


def parse_init_code(spec, n_expected):
    """Parse comma/space separated values, allowing x/?/. as unknowns."""
    parts = spec.replace(",", " ").split()
    if len(parts) != n_expected:
        raise ValueError(f"--init-code expected {n_expected} values, got {len(parts)}")
    values = []
    known = []
    for part in parts:
        if part.lower() in {"x", "?", "."}:
            values.append(None)
            known.append(False)
        else:
            values.append(int(part))
            known.append(True)
    return values, known


def apply_code_init(code_params, mode, init_values, init_strength, emb_w, valid_masks):
    """Initialize code params toward selected discrete values."""
    known_mask = torch.zeros(code_params.shape[0], dtype=torch.bool, device=code_params.device)
    with torch.no_grad():
        for i, value in enumerate(init_values):
            if value is None:
                continue
            tok = value_to_bytes(value)[0]
            if valid_masks is not None and not valid_masks[i][tok]:
                raise ValueError(f"Init value {value} at relative cell {i} violates constraints")
            if mode == "latent":
                code_params[i].copy_(emb_w[tok])
            else:
                code_params[i].fill_(-init_strength)
                code_params[i, tok] = init_strength
            known_mask[i] = True
    return known_mask


def build_nobranch_pcs(inp_batch, k, device):
    """Build (B, k) tensor of byte-encoded PC tokens for non-branching programs.

    For non-branching SUBLEQ, each step advances PC by 3.
    Entry s holds the PC at the START of step s.
    """
    B = inp_batch.shape[0]
    pcs = torch.zeros(B, k, dtype=torch.long, device=device)
    for b in range(B):
        pc_val = bytes_to_value([inp_batch[b, 0].item()])
        for s in range(k):
            pcs[b, s] = value_to_bytes(pc_val + s * 3)[0]
    return pcs


def sym_acc_multi(pred_code, pairs, code_offset, n_code_opt, k, no_branch=False):
    """Symbolic accuracy: inject pred_code, run k steps, compare final state.

    When no_branch=True, fixes every c operand (every 3rd cell starting at offset 2)
    to pc+3 before running, and only compares data memory (not PC), since branching
    behavior is excluded from the learning objective.
    """
    correct = 0
    for _, _, mem_i, pc_i, mem_f, pc_f in pairs:
        tm = list(mem_i)
        tm[code_offset:code_offset + n_code_opt] = list(pred_code)
        if no_branch:
            for i in range(2, n_code_opt, 3):
                tm[code_offset + i] = code_offset + i + 1
        m, p = tm, pc_i
        try:
            for _ in range(k):
                m, p, halted = step(m, p)
                if halted:
                    break
        except Exception:
            continue
        if no_branch:
            if m[DATA_START:] == mem_f[DATA_START:]:
                correct += 1
        else:
            if p == pc_f and m[DATA_START:] == mem_f[DATA_START:]:
                correct += 1
    return correct, len(pairs)


def main():
    ap = argparse.ArgumentParser(
        description="Multi-step program synthesis via chained differentiable execution")
    ap.add_argument("--checkpoint", default="round2_trained/checkpoints/best_model.pt")
    ap.add_argument("--exec-steps", type=int, default=8,
                    help="Number of execution steps to chain")
    ap.add_argument("--steps", type=int, default=2000, help="Optimization steps")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tau-start", type=float, default=5.0)
    ap.add_argument("--tau-end", type=float, default=0.1)
    ap.add_argument("--mode", choices=["gumbel", "gumbel_hard", "softmax", "latent"],
                    default="gumbel")
    ap.add_argument("--window-mode", choices=["any", "entry"], default="any",
                    help="Use all contiguous k-step windows or only the trace entrypoint")
    ap.add_argument("--state-mode", choices=["ste", "soft", "hard", "gumbel", "gumbel_hard"], default="ste",
                    help="How predicted intermediate states are fed into the next rollout step")
    ap.add_argument("--state-tau", type=float, default=1.0,
                    help="Temperature for soft intermediate state distributions")
    ap.add_argument("--loss-scope", choices=["final", "all", "curriculum"], default="final",
                    help="'final': only final state. 'all': every step. "
                         "'curriculum': blend all→final over training.")
    ap.add_argument("--sample-mode", choices=["uniform", "path", "start_pc"], default="uniform",
                    help="How to subsample candidate windows once traces are collected")
    ap.add_argument("--overcollect", type=int, default=8,
                    help="Candidate window multiplier before balanced subsampling")
    ap.add_argument("--code-cells", type=str, default="all")
    ap.add_argument("--code-offset", type=int, default=0)
    ap.add_argument("--n-io", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--init-code", type=str, default=None,
                    help="Comma/space separated initial code values; use x for unknown cells")
    ap.add_argument("--init-strength", type=float, default=20.0,
                    help="Logit strength used by --init-code for non-latent modes")
    ap.add_argument("--freeze-init", action="store_true",
                    help="Freeze the positions specified by --init-code")
    ap.add_argument("--pc", type=int, nargs="+", default=None,
                    help="Only keep windows whose executed PC path includes one of these PCs")
    ap.add_argument("--start-pc", type=int, nargs="+", default=None,
                    help="Only keep windows whose initial PC is one of these values")
    ap.add_argument("--end-pc", type=int, nargs="+", default=None,
                    help="Only keep windows whose last executed PC in the window is one of these values")
    ap.add_argument("--program",
                    choices=["negate", "addition", "countdown", "multiply",
                             "fibonacci", "div", "isqrt", "chain", "random", "random_safe"],
                    default="fibonacci")
    ap.add_argument("--chain-length", type=int, default=None,
                    help="Number of instructions for chain program (default: exec-steps)")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--data-only", action="store_true",
                    help="Tight masks: a,b in data cells (24-31), "
                         "c in instruction-aligned jumps + halt")
    ap.add_argument("--no-branch", action="store_true",
                    help="Assume non-branching program: hard-code intermediate PCs "
                         "and exclude PC from loss")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    k = args.exec_steps
    chain_length = args.chain_length or k
    device = torch.device(args.device if args.device != "auto" else auto_device())
    torch.manual_seed(args.seed)

    if args.checkpoint.lower() == "none":
        model = random_executor(device)
    else:
        print(f"Loading executor from {args.checkpoint}")
        model = load_executor(args.checkpoint, device)
    print(f"Model: {model.count_params():,} params (frozen)")

    # Code cells to optimize
    n_code_opt = CODE_SIZE if args.code_cells == "all" else int(args.code_cells)
    code_offset = args.code_offset
    code_positions = list(range(1 + code_offset, 1 + code_offset + n_code_opt))
    pc_filter = set(args.pc) if args.pc is not None else None
    start_pc_filter = set(args.start_pc) if args.start_pc is not None else None
    end_pc_filter = set(args.end_pc) if args.end_pc is not None else None

    # Generate multi-step I/O pairs
    rng = random.Random(args.seed)
    pairs_raw, first_mem, start_pc, n_train_candidates = collect_windows(
        args.program, rng, args.n_io, k, args.window_mode, args.sample_mode,
        pc_filter=pc_filter, start_pc_filter=start_pc_filter,
        end_pc_filter=end_pc_filter, overcollect=args.overcollect,
        chain_length=chain_length)
    if not pairs_raw:
        raise RuntimeError("Could not collect any training windows with the current filters")
    io_pairs = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in pairs_raw]
    step_tgt_batch = torch.stack([p[7] for p in pairs_raw]).to(device)
    true_code = first_mem[code_offset:code_offset + n_code_opt]

    # Print program info
    print(f"\nMulti-step synthesis: k={k} chained execution steps")
    print(f"Program: {args.program}", end="")
    if args.pc is not None:
        print(f"  (filtered to pc={sorted(args.pc)})", end="")
    if args.start_pc is not None:
        print(f"  (start_pc={sorted(args.start_pc)})", end="")
    if args.end_pc is not None:
        print(f"  (end_pc={sorted(args.end_pc)})", end="")
    print()
    print(f"Ground truth code cells mem[{code_offset}..{code_offset + n_code_opt - 1}]:")
    for i in range(0, n_code_opt, 3):
        abs_i = code_offset + i
        chunk = true_code[i:i+3]
        print(f"  instr @{abs_i}: ({', '.join(str(v) for v in chunk)})")

    # Show example traces
    print(f"\n{len(io_pairs)} training windows (k={k} steps, window_mode={args.window_mode}, sample_mode={args.sample_mode}):")
    if args.sample_mode != "uniform":
        print(f"  selected from {n_train_candidates} candidate windows")
    for idx in range(min(5, len(pairs_raw))):
        _, _, mi, pi, mf, pf, tpcs, _ = pairs_raw[idx]
        path = "->".join(str(p) for p in tpcs)
        print(f"  [{idx}] pc: {path}")
        print(f"       data in:  {mi[DATA_START:]}")
        print(f"       data out: {mf[DATA_START:]}")
    start_pc_counts = {}
    path_counts = {}
    for _, _, _, pi, _, _, _, _ in pairs_raw:
        start_pc_counts[pi] = start_pc_counts.get(pi, 0) + 1
    for _, _, _, _, _, _, pcs, _ in pairs_raw:
        path_counts[tuple(pcs)] = path_counts.get(tuple(pcs), 0) + 1
    pcs_summary = " ".join(f"@{pc}:{count}" for pc, count in sorted(start_pc_counts.items()))
    print(f"  start pc coverage: {pcs_summary}")
    if len(path_counts) <= 12:
        path_summary = " | ".join(
            f"{'->'.join(str(p) for p in path)}:{count}"
            for path, count in sorted(path_counts.items()))
        print(f"  path coverage: {path_summary}")

    # Generate test pairs
    test_pairs = []
    test_inp = test_tgt = None
    if args.n_test > 0:
        test_rng = random.Random(args.seed + 9999)
        test_raw, _, _, n_test_candidates = collect_windows(
            args.program, test_rng, args.n_test, k, args.window_mode, args.sample_mode,
            pc_filter=pc_filter, start_pc_filter=start_pc_filter,
            end_pc_filter=end_pc_filter, first_mem=first_mem,
            start_pc=start_pc, overcollect=args.overcollect,
            chain_length=chain_length)
        test_pairs = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in test_raw]
        if test_pairs:
            test_inp = torch.stack([p[0] for p in test_pairs]).to(device)
            test_tgt = torch.stack([p[1] for p in test_pairs]).to(device)
        print(f"{len(test_pairs)} held-out test I/O pairs", end="")
        if args.sample_mode != "uniform":
            print(f" (from {n_test_candidates} candidates)")
        else:
            print()
    test_pcs = build_nobranch_pcs(test_inp, k, device) if (args.no_branch and test_inp is not None) else None

    # Tensors
    inp_batch = torch.stack([p[0] for p in io_pairs]).to(device)
    tgt_batch = torch.stack([p[1] for p in io_pairs]).to(device)

    # Intermediate PCs and fixed code positions for non-branching mode
    train_pcs = build_nobranch_pcs(inp_batch, k, device) if args.no_branch else None
    code_pos_set = set(code_positions)
    all_code_tok_positions = list(range(1, 1 + CODE_SIZE))  # token positions 1..24
    fixed_positions = [p for p in all_code_tok_positions if p not in code_pos_set] if args.no_branch else None

    # Sanity check: chain with ground truth code
    gt_toks = [value_to_bytes(v)[0] for v in true_code]
    if args.no_branch:
        data_positions = list(range(1 + CODE_SIZE, SEQ_LEN))
    else:
        data_positions = [0] + list(range(1 + CODE_SIZE, SEQ_LEN))
    with torch.no_grad():
        logits = chain_discrete(model, inp_batch, gt_toks, code_positions, k,
                                intermediate_pcs=train_pcs,
                                fixed_positions=fixed_positions)
        preds = logits[:, data_positions].argmax(-1)
        tgts = tgt_batch[:, data_positions]
        n_ok = (preds == tgts).all(dim=1).sum().item()
    print(f"\nChain sanity (GT code, k={k}): {n_ok}/{len(io_pairs)} correct", end="")
    if n_ok < len(io_pairs):
        # Per-position accuracy for diagnostics
        per_pos = (preds == tgts).float().mean(dim=0)
        worst = per_pos.min().item()
        print(f"  (worst position: {worst:.1%})")
    else:
        print()

    # Constraint masks
    if args.data_only:
        valid_masks = make_data_only_masks(n_code_opt, device, code_offset)
        print(f"Data-only masks: a,b in {DATA_ADDR_TOKENS}, c in {INSTR_ALIGNED_TOKENS}")
    elif args.constrained:
        valid_masks = make_valid_masks(n_code_opt, device, code_offset)
        print(f"Constrained: a,b in [0,{MEM_SIZE-1}], c also allows -1")
    else:
        valid_masks = None

    # Learnable parameters
    emb_w = model.token_emb.weight
    if args.mode == "latent":
        code_params = nn.Parameter(torch.randn(n_code_opt, model.d_model, device=device) * 0.02)
    else:
        code_params = nn.Parameter(torch.zeros(n_code_opt, VOCAB_SIZE, device=device))
    freeze_mask = torch.zeros(n_code_opt, dtype=torch.bool, device=device)
    init_known = []
    if args.init_code is not None:
        init_values, init_known = parse_init_code(args.init_code, n_code_opt)
        known_mask = apply_code_init(
            code_params, args.mode, init_values, args.init_strength, emb_w, valid_masks)
        if args.freeze_init:
            freeze_mask = known_mask
    optimizer = torch.optim.Adam([code_params], lr=args.lr)

    tag = "+constrained" if args.constrained else ""
    print(f"\nOptimizing {n_code_opt} code cells over {args.steps} steps")
    print(f"  mode={args.mode}{tag}, chain={k} steps")
    print(f"  state_mode={args.state_mode}, loss_scope={args.loss_scope}")
    if args.no_branch:
        print(f"  no_branch: intermediate PCs hard-coded, PC excluded from loss")
    if args.init_code is not None:
        print(f"  init_code: {sum(init_known)}/{n_code_opt} cells specified, freeze_init={args.freeze_init}")
    if args.mode != "latent":
        print(f"  tau: {args.tau_start} -> {args.tau_end}")
    if args.no_branch:
        pos_labels = [f"m{i}" for i in range(CODE_SIZE, MEM_SIZE)]
    else:
        pos_labels = ["PC"] + [f"m{i}" for i in range(CODE_SIZE, MEM_SIZE)]
    print(f"  loss on {len(data_positions)} positions: {pos_labels}")
    print()

    # Training loop
    for step_i in range(1, args.steps + 1):
        frac = step_i / args.steps
        tau = args.tau_start * (args.tau_end / args.tau_start) ** frac

        s_tau = tau if args.state_mode.startswith("gumbel") else args.state_tau
        logits_steps = chain_forward(
            model, inp_batch, code_params, code_positions,
            k, tau, mode=args.mode, valid_masks=valid_masks,
            state_mode=args.state_mode, state_tau=s_tau, return_all=True,
            intermediate_pcs=train_pcs, fixed_positions=fixed_positions)
        loss_final = F.cross_entropy(
            logits_steps[-1][:, data_positions].reshape(-1, VOCAB_SIZE),
            tgt_batch[:, data_positions].reshape(-1))
        if args.loss_scope == "final":
            loss = loss_final
        else:
            losses_all = []
            for t in range(k):
                losses_all.append(F.cross_entropy(
                    logits_steps[t][:, data_positions].reshape(-1, VOCAB_SIZE),
                    step_tgt_batch[:, t][:, data_positions].reshape(-1)))
            loss_all = sum(losses_all) / len(losses_all)
            if args.loss_scope == "all":
                loss = loss_all
            else:
                alpha = frac  # curriculum: 0 (all) → 1 (final)
                loss = (1 - alpha) * loss_all + alpha * loss_final

        optimizer.zero_grad()
        loss.backward()
        if freeze_mask.any():
            code_params.grad[freeze_mask] = 0
        optimizer.step()

        if step_i % 100 == 0 or step_i == 1:
            with torch.no_grad():
                if args.mode == "latent":
                    guesses = decode_latent(code_params, emb_w, valid_masks)
                else:
                    guesses = _masked_argmax(code_params, valid_masks)
                decoded = [bytes_to_value([g]) for g in guesses]
                n_match = sum(a == b for a, b in zip(decoded, true_code))

                tau_s = f" | tau={tau:.3f}" if args.mode != "latent" else ""

                # Neural accuracy (discrete chain)
                out_tr = chain_discrete(model, inp_batch, guesses, code_positions, k,
                                        intermediate_pcs=train_pcs,
                                        fixed_positions=fixed_positions)
                n_tr = (out_tr[:, data_positions].argmax(-1) == tgt_batch[:, data_positions]).all(1).sum().item()

                # Symbolic accuracy
                s_tr_c, s_tr_t = sym_acc_multi(decoded, io_pairs, code_offset, n_code_opt, k, no_branch=args.no_branch)

                line = (f"step {step_i:5d} | loss={loss.item():.4f}{tau_s} | "
                        f"{n_match}/{n_code_opt} cells | "
                        f"neural={n_tr}/{len(io_pairs)}({n_tr*100//len(io_pairs)}%) "
                        f"sym={s_tr_c}/{s_tr_t}({s_tr_c*100//s_tr_t}%)")

                if test_inp is not None:
                    out_te = chain_discrete(model, test_inp, guesses, code_positions, k,
                                            intermediate_pcs=test_pcs,
                                            fixed_positions=fixed_positions)
                    n_te = (out_te[:, data_positions].argmax(-1) == test_tgt[:, data_positions]).all(1).sum().item()
                    s_te_c, s_te_t = sym_acc_multi(decoded, test_pairs, code_offset, n_code_opt, k, no_branch=args.no_branch)
                    line += (f" | test: neural={n_te}/{len(test_pairs)}({n_te*100//len(test_pairs)}%) "
                             f"sym={s_te_c}/{s_te_t}({s_te_c*100//s_te_t}%)")

                print(line)

    # Final results
    with torch.no_grad():
        if args.mode == "latent":
            final_toks = decode_latent(code_params, emb_w, valid_masks)
        else:
            final_toks = _masked_argmax(code_params, valid_masks)
    pred_code = [bytes_to_value([t]) for t in final_toks]

    hdr = f"{'Instr':>10} | {'Recovered':>20} | {'Target':>20} | Match"
    print(f"\n{'='*len(hdr)}")
    print(hdr)
    print(f"{'-'*len(hdr)}")
    for i in range(0, n_code_opt, 3):
        pred_chunk = pred_code[i:i+3]
        true_chunk = true_code[i:i+3]
        match = pred_chunk == true_chunk
        abs_i = code_offset + i
        print(f"{'@'+str(abs_i):>10} | {str(tuple(pred_chunk)):>20} | {str(tuple(true_chunk)):>20} | {'yes' if match else 'NO':>5}")

    n_diff = sum(a != b for a, b in zip(pred_code, true_code))
    if n_diff == 0:
        print(f"\nEXACT MATCH - all {n_code_opt} cells recovered")
    else:
        print(f"\n{n_diff}/{n_code_opt} cells differ from ground truth")

    with torch.no_grad():
        out_tr = chain_discrete(model, inp_batch, final_toks, code_positions, k,
                                intermediate_pcs=train_pcs,
                                fixed_positions=fixed_positions)
        n_tr = (out_tr[:, data_positions].argmax(-1) == tgt_batch[:, data_positions]).all(1).sum().item()
        s_tr_c, s_tr_t = sym_acc_multi(pred_code, io_pairs, code_offset, n_code_opt, k, no_branch=args.no_branch)
        print(f"\nTrain (k={k})  neural: {n_tr}/{len(io_pairs)} ({n_tr*100//len(io_pairs)}%)  "
              f"symbolic: {s_tr_c}/{s_tr_t} ({s_tr_c*100//s_tr_t}%)")
        if test_inp is not None:
            out_te = chain_discrete(model, test_inp, final_toks, code_positions, k,
                                    intermediate_pcs=test_pcs,
                                    fixed_positions=fixed_positions)
            n_te = (out_te[:, data_positions].argmax(-1) == test_tgt[:, data_positions]).all(1).sum().item()
            s_te_c, s_te_t = sym_acc_multi(pred_code, test_pairs, code_offset, n_code_opt, k, no_branch=args.no_branch)
            print(f"Test  (k={k})  neural: {n_te}/{len(test_pairs)} ({n_te*100//len(test_pairs)}%)  "
                  f"symbolic: {s_te_c}/{s_te_t} ({s_te_c*100//s_te_t}%)")


if __name__ == "__main__":
    main()
