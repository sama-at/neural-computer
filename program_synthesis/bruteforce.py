#!/usr/bin/env python3
"""Brute-force program search: enumerate all candidate programs and check against I/O pairs.

Usage:
    python bruteforce.py --program chain --exec-steps 2 --code-cells 6
    python bruteforce.py --program chain --exec-steps 2 --code-cells 6 --restrict
"""

import sys
import os
import random
import argparse
import itertools
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'round2_trained'))

from subleq import step, encode, value_to_bytes, bytes_to_value
from subleq.interpreter import MEM_SIZE, VOCAB_SIZE, CODE_SIZE, DATA_START

sys.path.insert(0, os.path.dirname(__file__))
from synthesize import make_program_state


def make_io_pairs(program, k, n_io, seed):
    """Generate n_io input/output pairs by running the symbolic interpreter for k steps."""
    rng = random.Random(seed)
    pairs = []
    first_mem = None
    for _ in range(n_io * 10):
        if len(pairs) >= n_io:
            break
        mem, pc = make_program_state(program, rng)
        if first_mem is None:
            first_mem = list(mem)
        m, p = list(mem), pc
        try:
            for _ in range(k):
                m, p, halted = step(m, p)
                if halted:
                    break
        except Exception:
            continue
        pairs.append((list(mem), pc, list(m), p))
    return pairs, first_mem


def check_candidate(candidate, pairs, code_offset, k):
    """Check if candidate code produces correct output for all I/O pairs."""
    n_code = len(candidate)
    correct = 0
    for mem_i, pc_i, mem_f, pc_f in pairs:
        tm = list(mem_i)
        tm[code_offset:code_offset + n_code] = list(candidate)
        m, p = tm, pc_i
        try:
            for _ in range(k):
                m, p, halted = step(m, p)
                if halted:
                    break
        except Exception:
            continue
        if p == pc_f and m[DATA_START:] == mem_f[DATA_START:]:
            correct += 1
    return correct


def main():
    ap = argparse.ArgumentParser(description="Brute-force program search")
    ap.add_argument("--program",
                    choices=["negate", "addition", "countdown", "multiply",
                             "fibonacci", "div", "isqrt", "chain", "random", "random_safe"],
                    default="chain")
    ap.add_argument("--exec-steps", type=int, default=2, help="Number of execution steps (k)")
    ap.add_argument("--code-cells", type=int, default=6, help="Number of code cells to search")
    ap.add_argument("--code-offset", type=int, default=0)
    ap.add_argument("--n-io", type=int, default=50, help="Number of I/O pairs")
    ap.add_argument("--restrict", action="store_true",
                    help="Restrict search: a,b in [DATA_START..MEM_SIZE-1], "
                         "c in {-1, 0, 3, 6, ..., 21}")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    k = args.exec_steps
    n_code = args.code_cells
    code_offset = args.code_offset

    # Build value domains for each cell position
    if args.restrict:
        a_vals = list(range(DATA_START, MEM_SIZE))    # 24..31 (8 values)
        b_vals = list(range(DATA_START, MEM_SIZE))    # 24..31 (8 values)
        c_vals = [-1] + list(range(0, CODE_SIZE, 3))  # -1, 0, 3, 6, ..., 21 (9 values)
    else:
        a_vals = list(range(MEM_SIZE))                 # 0..31 (32 values)
        b_vals = list(range(MEM_SIZE))                 # 0..31 (32 values)
        c_vals = [-1] + list(range(MEM_SIZE))          # -1, 0, ..., 31 (33 values)

    # Build per-cell domains
    domains = []
    for i in range(n_code):
        pos_in_instr = (code_offset + i) % 3
        if pos_in_instr == 0:
            domains.append(a_vals)
        elif pos_in_instr == 1:
            domains.append(b_vals)
        else:
            domains.append(c_vals)

    total_search = 1
    for d in domains:
        total_search *= len(d)
    per_cell_sizes = [len(d) for d in domains]

    print(f"Brute-force search: {n_code} cells, k={k} steps")
    print(f"Restrict: {args.restrict}")
    print(f"Per-cell domain sizes: {per_cell_sizes}")
    print(f"Total search space: {total_search:,}")

    if total_search > 1_000_000_000:
        print("WARNING: search space > 1B, this will take a very long time")

    # Generate I/O pairs
    pairs, first_mem = make_io_pairs(args.program, k, args.n_io, args.seed)
    true_code = first_mem[code_offset:code_offset + n_code]
    print(f"\nGround truth: {true_code}")
    print(f"Generated {len(pairs)} I/O pairs")

    # Verify ground truth
    gt_correct = check_candidate(true_code, pairs, code_offset, k)
    print(f"Ground truth score: {gt_correct}/{len(pairs)}")

    # Search
    t0 = time.time()
    checked = 0
    perfect = []
    best_score = 0
    report_interval = max(1, total_search // 20)

    for candidate in itertools.product(*domains):
        score = check_candidate(candidate, pairs, code_offset, k)
        checked += 1

        if score > best_score:
            best_score = score
            print(f"  new best: {list(candidate)} -> {score}/{len(pairs)}")

        if score == len(pairs):
            perfect.append(list(candidate))
            if len(perfect) <= 10:
                match = list(candidate) == true_code
                print(f"  PERFECT: {list(candidate)} {'(GT)' if match else ''}")

        if checked % report_interval == 0:
            elapsed = time.time() - t0
            rate = checked / elapsed
            eta = (total_search - checked) / rate if rate > 0 else 0
            print(f"  progress: {checked:,}/{total_search:,} ({100*checked/total_search:.1f}%) "
                  f"  {rate:.0f}/s  ETA {eta:.0f}s  perfect={len(perfect)}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({checked:,} candidates)")
    print(f"Perfect solutions: {len(perfect)}")
    if perfect:
        for i, p in enumerate(perfect[:20]):
            match = p == true_code
            tag = " (GT)" if match else ""
            print(f"  [{i}] {p}{tag}")
        if len(perfect) > 20:
            print(f"  ... and {len(perfect) - 20} more")
        gt_found = any(p == true_code for p in perfect)
        print(f"Ground truth in solutions: {gt_found}")


if __name__ == "__main__":
    main()
