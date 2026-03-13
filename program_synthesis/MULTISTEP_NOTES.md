# Multistep Synthesis Notes

## Goal

Make a stronger demo than single-step instruction recovery by chaining the learned SUBLEQ executor for multiple steps and optimizing code through the unrolled computation.

## Current Script

`program_synthesis/synthesize_multistep.py`

## Findings So Far

- `k=1` works trivially but is not meaningful when examples always start at the program entrypoint. For `fibonacci`, every example begins at `pc=0`, so the optimizer only needs to match the first step.
- `k=1` recovered highly non-symbolic code while still getting `100%` train/test accuracy. This is the same underdetermination issue as the single-step script.
- `k=2` also reached `100%` train/test on `fibonacci`, but still recovered mostly wrong code. Two-step windows from the entrypoint are still too weak.
- `k=4` and `k=8` failed completely with the first chained implementation. Loss decreased somewhat, but train/test stayed at `0%`.

## Likely Reasons

- The current dataset uses only windows that start from the initial program state, so it undercovers the space of reachable program counters.
- For `fibonacci`, the `@18` halt branch is often not observed in the current windows, so `c=-1` is still weakly identified.
- Gradients through long chains are likely too weak/noisy with the current straight-through state propagation.

## Next Changes

- Sample arbitrary contiguous `k`-step windows from full execution traces instead of always starting at the entrypoint.
- Allow windows that end in halt, so branch-to-halt behavior is visible.
- Try softer intermediate state propagation if the straight-through rollout still collapses for `k>=4`.
- If needed, add optional auxiliary loss on intermediate states for stability.

## Implemented

- `synthesize_multistep.py` now samples contiguous `k`-step windows from full traces with `--window-mode any|entry`.
- Added `--state-mode ste|soft|hard` to control how intermediate predicted states are rolled forward.
- Added `--loss-scope final|all` so I can compare pure final-state supervision against auxiliary supervision on every step in the window.

## Current Experiments Running

- `k=4`, `window_mode=any`, `state_mode=ste`, `loss_scope=final`
- `k=4`, `window_mode=any`, `state_mode=soft`, `loss_scope=final`
- `k=4`, `window_mode=any`, `state_mode=soft`, `loss_scope=all`

## Results So Far

- Full-program `k=4`:
  - `state_mode=ste`, `loss_scope=final` stayed near `0-2%`.
  - `state_mode=soft`, `loss_scope=final` stayed at `0%`.
  - `state_mode=soft`, `loss_scope=all` reached `25%` train/test and recovered several correct cells. This is the first full-program setting that clearly beats collapse.
- Targeted multistep optimization is much stronger:
  - `k=4`, `code_offset=12`, `code_cells=9`, `state_mode=soft`, `loss_scope=all` reached `97-98%` by about step `300`.
  - `k=4`, `code_offset=18`, `code_cells=3`, `state_mode=soft`, `loss_scope=all` also reached `97-98%` by about step `300`.
- This is encouraging because the `@18` subset includes the hard branch-to-halt instruction. Single-step optimization struggled there; unrolled multistep windows make it learnable.
- The remaining miss on `@18.c` looks like class imbalance, not pure identifiability failure. In sampled `k=4` fibonacci windows that include `@18`, only about `500 / 9540 ~= 5%` actually take the halt branch to `-1`; the other `95%` go to `21`. So a model can be near-perfect on behavior while still guessing the rare branch target incorrectly.

## New Changes

- Added `--sample-mode uniform|path|start_pc` plus `--overcollect` so I can collect a larger candidate pool and then rebalance which windows are actually used.
- Added `--init-code` and `--freeze-init` so I can warm-start known cells and isolate the genuinely unresolved ones.
- Added `--end-pc` to keep only windows whose final executed instruction is a chosen PC. This is important because a branch target only directly affects the final supervised state when that branch instruction is the last executed step in the window.

## New Results

- Path-balancing all `k=4` windows was informative but not sufficient:
  - `code_offset=18`, `code_cells=3`, `sample_mode=path`, `state_mode=soft`, `loss_scope=all` dropped from the earlier `97-98%` to about `88-89%`.
  - This is actually useful evidence, not a pure regression: with path-balanced data, the only remaining error is the halt path. The recovered instruction stayed `(30, 29, 2)`, so it fails exactly on the balanced `9->12->15->18->-1` windows.
- Optimizer choice on that same balanced dataset:
  - `mode=softmax` stayed stuck at about `44-45%`.
  - `mode=latent` drove the continuous loss near zero while discrete accuracy stayed `44-45%`, which is strong evidence of a soft/adversarial continuous solution rather than real symbolic recovery.
- The key fix was targeting supervision more precisely:
  - `--end-pc 18 --sample-mode path --init-code 30,29,x --freeze-init`
  - This produces exactly two balanced window types: `9->12->15->18->-1` and `9->12->15->18->21`.
  - In that setting, both `gumbel` and `softmax` recover `@18 = (30, 29, -1)` exactly and reach `200/200` train and `200/200` test on `k=4`.

## Current Take

- The earlier `@18` failure was not "multistep cannot learn the branch"; it was "the dataset mostly did not place the branch under direct supervision."
- `--pc 18` was too broad because many windows merely *contain* `18`; they do not *end with* execution of `18`, so the branch target is not directly visible at the final supervised state.
- `--end-pc` seems like the right abstraction for staged recovery: solve the cells whose effects are actually exposed at the end of the rollout, then warm-start a broader optimization.

## Next Steps

- Try the same `--end-pc` trick for other hard cells/instructions to build a better staged warm start for the full `24`-cell `k=4` run.
- Check which instructions are behaviorally underdetermined even with `k=4`, so I do not waste time chasing an exact symbolic match where only behavioral recovery is identifiable.

## Follow-Up Results

- `@21` behaves like `@18`: `--end-pc 21 --sample-mode path` recovers it exactly and reaches `200/200` train and test on `k=4`.
- The early block does **not** work well as one joint chunk:
  - `code_offset=0`, `code_cells=12`, `sample_mode=path` only reached about `37%`.
  - So the staged approach still matters; broad joint recovery from scratch is too entangled.
- Individual end-of-window solves for the front half:
  - `@3` with `--end-pc 3` recovers exactly.
  - `@9` with `--end-pc 9` recovers exactly.
  - `@0` with `--end-pc 0` reaches `200/200`, but with `(23, 13, 3)` instead of `(31, 31, 3)`.
  - `@6` with `--end-pc 6` reaches `200/200`, but with `(31, 27, 30)` instead of `(31, 27, 9)`.
- This is strong evidence that some instruction cells are behaviorally underdetermined under the available supervision, even when the overall `k=4` behavior is fully correct.

## Composed Full Program

- I took the best staged instruction-level recoveries:
  - `@0 = (23, 13, 3)`
  - `@3 = (28, 31, 6)`
  - `@6 = (31, 27, 30)`
  - `@9 = (31, 31, 12)`
  - `@12 = (27, 31, 15)`
  - `@15 = (31, 28, 5)`
  - `@18 = (30, 29, -1)`
  - `@21 = (31, 31, 0)`
- Freezing that composed `24`-cell init on the full `k=4` path-balanced objective gives:
  - `200/200` train neural
  - `200/200` train symbolic
  - `200/200` test neural
  - `200/200` test symbolic
- So the staged multistep procedure now succeeds on the full-program behavioral objective for `k=4`.
- However, the composed program is only `20/24` cells identical to the ground-truth code. The mismatches are exactly the kind of cells that now look behaviorally underdetermined rather than merely hard to optimize.

## Current Conclusion

- I now have a real `k=4` success case for the proof-of-concept:
  - Multistep chaining plus targeted supervision (`--end-pc`) can recover the hard branch instruction exactly.
  - Staged recovery of locally behaviorally-correct instructions can be composed into a full program that gets perfect held-out `k=4` behavior.
- The remaining gap is no longer "cannot get to `100%` behavior." I can.
- The remaining gap is "exact symbolic identity is not always identifiable from this supervision," which is an important result in its own right.
