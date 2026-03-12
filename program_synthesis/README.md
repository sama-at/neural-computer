# Program Synthesis by Backpropagating Through a Learned Executor

## Idea

The Round 2 SUBLEQ transformer is a differentiable function: 33 tokens in → 33 tokens out, performing one SUBLEQ step. Because the executor lives inside the network (not behind a tool call), we can backpropagate *through* it to optimize the **input**.

Specifically: fix the data cells and optimize the code cells to produce a desired output after **one single step**. This is program synthesis via gradient descent — made possible only because execution is internalized.

## Scope: Single-Step Only

No iteration. No chaining steps. One forward pass through the trained executor.

Given:
- **Known inputs**: PC value + data cell values
- **Known output**: desired state after one SUBLEQ step
- **Unknown**: the code cells (the instruction)

Optimize the code cells so that `executor(state_with_code) → desired_output`.

## What One SUBLEQ Step Does (Recap)

```
a, b, c = mem[pc], mem[pc+1], mem[pc+2]
mem[b] -= mem[a]
if mem[b] <= 0: pc = c
else:           pc = pc + 3
```

So the "program" for one step is just three numbers: (a, b, c). Given a known starting state and a known ending state, find (a, b, c).

## Concrete Setup

### Round 2 model

- 32 memory cells, 8-bit values [-128, 127]
- Vocab: 256 tokens (byte-level, two's complement)
- Sequence: 33 tokens — `[pc, mem[0], mem[1], ..., mem[31]]`
- Architecture: 6 layers, 8 heads, d_model=256, 4.9M params
- Trained on single-step transitions to 100% accuracy

### The optimization problem

```
Input token sequence:  [pc, mem[0], mem[1], mem[2], ..., mem[31]]
                        ↑    ↑       ↑       ↑       ↑
                       fixed LEARNABLE (code)        fixed (data)

Output token sequence: [pc', mem'[0], ..., mem'[31]]  ← target (known)

Loss = cross_entropy(executor_output_logits, target_tokens)
Gradients flow back into the code cell token representations.
```

### Token representation problem

The executor does `self.token_emb(tokens)` — a discrete lookup. Can't differentiate through that for the learnable positions. Options:

**Option A: Gumbel-softmax**
Each learnable code cell is a 256-dim logit vector. Softmax (with temperature τ) gives a distribution over the 256 byte values. The embedding for that position is the weighted sum of all 256 embedding vectors. During optimization, anneal τ from high (soft) to low (hard). After convergence, argmax gives the discrete token.

**Option B: Straight-through estimator**
Keep a continuous "soft token" in [0, 255]. Round to nearest int for the forward pass. On the backward pass, pass gradients through as if rounding didn't happen.

**Option C: Optimize in embedding space directly**
Bypass the embedding table for learnable positions. Directly optimize a (d_model,) vector for each code position. After convergence, find the nearest embedding vector to decode back to a token.

**Recommendation: Start with Option A (Gumbel-softmax).** It's the most principled, gives a proper probability distribution we can inspect, and annealing is well-understood.

## Example: Recover a Negate Instruction

### Ground truth

The negate program's first instruction is: `SUBLEQ(25, 25, 3)` — meaning a=25, b=25, c=3. This clears mem[25] (subtracts it from itself) and branches to PC=3.

```
Before: pc=0, mem[0]=25, mem[1]=25, mem[2]=3, ..., mem[25]=X, ...
After:  pc=3, mem[0]=25, mem[1]=25, mem[2]=3, ..., mem[25]=0, ...
```

### What we give the optimizer

- **Fixed**: pc=0, mem[3..31] (data cells and remaining code)
- **Learnable**: mem[0], mem[1], mem[2] (the three operands of the instruction at pc=0)
- **Target**: the known output state after one step

### What we expect

The optimizer finds mem[0]=25, mem[1]=25, mem[2]=3 (or any equivalent instruction that produces the same output state).

## Difficulty Ladder

| Level | Task | Code cells | I/O pairs | Notes |
|-------|------|-----------|-----------|-------|
| 1 | Recover one known instruction | 3 (a, b, c) | 1 | Sanity check. The target is unique given a specific starting state. |
| 2 | Recover one instruction from multiple I/O pairs | 3 | 3-5 | Same instruction, different data values. Tests whether optimizer finds the *general* instruction, not one that overfits to a single state. |
| 3 | Recover one instruction with unknown branch target | 3 | 3-5 | c could be anything — tests whether gradient signal from the PC output is enough to pin it down. |
| 4 | Distinguish between programs | 3 | 5-10 | Given I/O pairs, can it distinguish "subtract mem[24] from mem[25]" vs "subtract mem[25] from mem[24]"? These differ by swapping a and b. |

## Implementation Plan

### Files

```
program_synthesis/
├── README.md           # This file
├── synthesize.py       # Main optimization script
└── (uses round2_trained/subleq/ for model + tokenizer)
```

### synthesize.py outline

```python
# 1. Load the trained executor (frozen)
model = load_checkpoint("round2_trained/checkpoints/best_model.pt")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# 2. Define the learnable code cells as Gumbel-softmax logits
#    For one instruction at pc=0: positions 1, 2, 3 in the token sequence
#    (position 0 is PC, positions 1-3 are mem[0], mem[1], mem[2])
code_logits = nn.Parameter(torch.zeros(3, 256))  # 3 cells × 256 vocab

# 3. For each I/O pair:
#    a. Build the input token sequence
#       - PC and data cells: hard tokens → embeddings via frozen embedding table
#       - Code cells: Gumbel-softmax(code_logits, τ) → weighted embedding sum
#    b. Run the executor forward (get output logits)
#    c. Compute loss against target output tokens

# 4. Optimize code_logits with Adam, annealing τ from 5.0 → 0.1

# 5. Read out: argmax(code_logits) → the discovered instruction
```

### Key detail: hybrid embedding construction

For each position in the 33-token input:
- If the position is **fixed** (PC, data cells): normal `token_emb(token_id)`
- If the position is **learnable** (code cells): `softmax(logits / τ) @ token_emb.weight` — a soft weighted sum over all 256 embeddings

This produces a (batch, 33, d_model) tensor that goes through the rest of the model normally. The model's weights are frozen; only `code_logits` gets gradients.

## Prerequisite

A trained Round 2 checkpoint. Either:
- `cd round2_trained && python train.py` (~2h on GPU)
- Or use a pre-existing checkpoint if available

## Success Criteria

**Level 1 passes if**: Given one I/O pair for the first step of negate(X), the optimizer recovers a=25, b=25, c=3 (or an equivalent instruction).

**Stretch**: Works for multiple I/O pairs simultaneously, finding the instruction that satisfies all of them.

## Why This Matters

If execution is a **tool call** (Python interpreter), you cannot do this. `subprocess.run("python interpreter.py")` has no gradients. The only way to synthesize programs would be brute-force search or LLM generation-and-test.

Because execution is **internalized** in a differentiable neural network, gradient descent can search the program space directly. The executor's learned weights encode the semantics of SUBLEQ, and those semantics are differentiable.
