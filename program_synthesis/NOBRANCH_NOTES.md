# No-Branch Multi-Step Synthesis Notes

Goal: Learn programs from input->output pairs only (no intermediate supervision).
Using `--loss-scope final` so only the final state after k steps is supervised.
Key change: `--no-branch` hard-codes intermediate PCs and excludes PC from loss.
Clean intermediate state: only data cells come from model output; PC and all code are injected.

## Bug Found: Chain Program Code Corruption

The chain program's @12 instruction (0,0,0) writes to mem[0], corrupting the code.
50% of chain programs loop (when mem[28]>0 after @9), and the loop corrupts mem[0] from 24 to 0.
Windows from subsequent loops have corrupted code that doesn't match `true_code`.
Fix: use `--window-mode entry` to only use windows from the first pass.

## Key Findings

### Hyperparameters that work
- `--state-mode ste` (hard forward, soft backward) — critical for k>=3. Soft mode fails.
- `--lr 0.1`, `--tau-start 2.0 --tau-end 0.05` — higher LR and lower starting tau vs original
- `--loss-scope curriculum` — smoothly blends all→final: `loss = (1-α)*loss_all + α*loss_final`, α=frac
- `--no-branch --data-only --window-mode entry` — standard for non-branching chain programs

### What's recoverable
- a,b operands: ALWAYS recovered (the data addresses are directly observable in I/O)
- c operands: NEVER recovered for non-branching instructions (c doesn't affect execution)
- c for the LAST instruction (halt): sometimes recovered (branching IS observable)

## Experiment History

### Exp 1-4: Initial exploration (see earlier notes)
Discovered clean intermediate states, data-only masks, code corruption fix.

### Exp 5: End-to-end k=2, loss-scope=final, state-mode=soft — FAILED
Loss stuck at 2.95, 0% accuracy. Chicken-and-egg problem.

### Exp 6: Warm-start a0,b0, k=2 — SUCCESS
`--init-code '24,25,x,x,x,x' --loss-scope final` → 100% by step 100.

### Exp 7: k=2 curriculum (all→final) from scratch — SUCCESS ★
`--loss-scope curriculum --state-mode soft` → 100% by step 300. No warm-start needed.

### Exp 8: k=3 with soft state mode — FAILED
Even loss-scope=all doesn't converge. Soft intermediates compound noise over 2+ steps.

### Exp 9: k=3 with STE + better hyperparams — SUCCESS ★
`--state-mode ste --lr 0.1 --tau-start 2.0 --tau-end 0.05` → 100% by step 100.
Key insight: STE gives discrete-looking intermediate inputs the model can process correctly.

### Exp 10: k=3 curriculum + STE — SUCCESS ★
Same hyperparams as Exp 9 with `--loss-scope curriculum` → 100% by step 100.

### Exp 11: k=4 curriculum + STE — SUCCESS ★
All 4 chain instructions recovered. @9=(27,28,-1) is exact match.
Neural: 199/200 train (GT sanity 198/200 due to model imprecision), 200/200 test.
Symbolic: 200/200 train, 200/200 test.

## Summary of Results

| k | code cells | loss-scope | neural train | sym train | sym test |
|---|-----------|------------|-------------|-----------|----------|
| 2 | 6         | curriculum | 200/200     | 200/200   | 200/200  |
| 3 | 9         | curriculum | 200/200     | 200/200   | 200/200  |
| 4 | 12        | curriculum | 199/200     | 200/200   | 200/200  |

Working recipe: `--loss-scope curriculum --state-mode ste --no-branch --data-only
--lr 0.1 --tau-start 2.0 --tau-end 0.05 --window-mode entry --start-pc 0`
