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
    MiniSUBLEQTransformer, make_negate, step, encode, decode,
    value_to_bytes, bytes_to_value,
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
      gumbel:  code_params is (N, vocab) logits → Gumbel-softmax → weighted embedding sum
      softmax: code_params is (N, vocab) logits → softmax(logits/τ) → weighted embedding sum
      latent:  code_params is (N, d_model) vectors injected directly as embeddings
    """
    B, S = fixed_tokens.shape

    tok_emb = model.token_emb(fixed_tokens).clone()

    for i, pos in enumerate(code_positions):
        if mode == "latent":
            tok_emb[:, pos] = code_params[i].unsqueeze(0).expand(B, -1)
        else:
            logits_i = code_params[i].unsqueeze(0).expand(B, -1)
            if mode == "gumbel":
                probs = F.gumbel_softmax(logits_i, tau=tau, hard=False)
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


def make_io_pair(val):
    """Create input/target token pair for first step of negate(val)."""
    mem, pc, _ = make_negate(val)
    input_tokens = encode(mem, pc)
    new_mem, new_pc, _ = step(mem, pc)
    target_tokens = encode(new_mem, new_pc)
    return input_tokens, target_tokens, mem, pc, new_mem, new_pc


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
    parser.add_argument("--mode", choices=["gumbel", "softmax", "latent"], default="gumbel",
                        help="gumbel: Gumbel-softmax over vocab logits; "
                             "softmax: plain softmax with temperature; "
                             "latent: optimize directly in embedding space")
    parser.add_argument("--code-cells", type=str, default="3",
                        help="Number of code cells to optimize: integer or 'all' for full code region")
    parser.add_argument("--n-io", type=int, default=1, help="Number of I/O pairs to optimize over")
    parser.add_argument("--val", type=int, default=42, help="Input value for negate (used when --n-io=1)")
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

    # --- Create I/O pairs ---
    if args.n_io == 1:
        vals = [args.val]
    else:
        rng = random.Random(args.seed)
        vals = [rng.randint(-100, 100) for _ in range(args.n_io)]

    io_pairs = [make_io_pair(v) for v in vals]
    true_code = io_pairs[0][2][:n_code]

    print(f"\nGround truth code cells mem[0..{n_code-1}]:")
    for i in range(0, n_code, 3):
        chunk = true_code[i:i+3]
        print(f"  instr @{i}: ({', '.join(str(v) for v in chunk)})")
    print(f"\n{args.n_io} I/O pair(s):")
    for v, (_, _, mem, pc, new_mem, new_pc) in zip(vals, io_pairs):
        print(f"  val={v:4d}: pc {pc}->{new_pc}, mem[25] {mem[25]}->{new_mem[25]}")

    inp_batch = torch.stack([p[0] for p in io_pairs]).to(device)
    tgt_batch = torch.stack([p[1] for p in io_pairs]).to(device)

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

    # --- Optimization loop ---
    for step_i in range(1, args.steps + 1):
        frac = step_i / args.steps
        tau = args.tau_start * (args.tau_end / args.tau_start) ** frac

        out_logits = hybrid_forward(
            model, inp_batch, code_params, code_positions, tau, mode=args.mode)
        loss = F.cross_entropy(out_logits.view(-1, VOCAB_SIZE), tgt_batch.view(-1))

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
                print(f"step {step_i:5d} | loss={loss.item():.4f}{tau_str} | "
                      f"{n_match}/{n_code} correct | {conf_str}"
                      f"{' *' if n_match == n_code else ''}")

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

    if all_match:
        print("\nEXACT MATCH")
    else:
        # Check semantic equivalence against all I/O pairs
        equiv = True
        for _, _, mem_i, pc_i, new_mem_i, new_pc_i in io_pairs:
            test_mem = list(mem_i)
            test_mem[:n_code] = pred_code
            result_mem, result_pc, _ = step(test_mem, 0)
            if result_mem != list(new_mem_i) or result_pc != new_pc_i:
                equiv = False
                break
        if equiv:
            print(f"\nEQUIVALENT — different code, same behavior on all {args.n_io} I/O pairs")
        else:
            n_diff = sum(a != b for a, b in zip(pred_code, true_code))
            print(f"\nFAILED — {n_diff}/{n_code} cells wrong")


if __name__ == "__main__":
    main()
