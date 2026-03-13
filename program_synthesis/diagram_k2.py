#!/usr/bin/env python3
"""Diagram showing forward information flow and backward gradient flow for k=2
no-branch multi-step synthesis."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, axes = plt.subplots(2, 1, figsize=(16, 18))

# Colors
C_INPUT = "#4ECDC4"
C_CODE = "#FF6B6B"
C_CODE_DET = "#FF6B6B"
C_MODEL = "#45B7D1"
C_STATE = "#96CEB4"
C_LOSS = "#FFEAA7"
C_PC = "#DDA0DD"
C_GRAD = "#FF4444"
C_DETACH = "#CCCCCC"
C_DATA = "#F0E68C"

def draw_box(ax, xy, w, h, label, color, fontsize=9, alpha=0.85, bold=False):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="black",
                                    linewidth=1.2, alpha=alpha)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(xy[0] + w/2, xy[1] + h/2, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True)

def draw_arrow(ax, start, end, color="black", lw=1.5, style="-|>", ls="-"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, color=color,
                            linewidth=lw, linestyle=ls,
                            mutation_scale=15, connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)

# ============================================================
# TOP: Forward pass
# ============================================================
ax = axes[0]
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Forward Pass (k=2, no-branch, loss-scope=final)", fontsize=14, fontweight="bold", pad=15)

# --- Learnable code params ---
draw_box(ax, (0.5, 9.0), 2.5, 0.8, "code_params[0:3]\n@0: (a₀,b₀,c₀)", C_CODE, fontsize=8, bold=True)
draw_box(ax, (4.0, 9.0), 2.5, 0.8, "code_params[3:6]\n@3: (a₁,b₁,c₁)", C_CODE, fontsize=8, bold=True)
ax.text(3.25, 10.1, "Learnable Parameters (Gumbel-Softmax → embeddings)", ha="center",
        fontsize=10, fontweight="bold", color=C_CODE)

# --- STEP 0 ---
ax.text(3.5, 8.2, "STEP 0  (PC=0, executes @0)", fontsize=11, fontweight="bold",
        ha="center", color="#333")

# Input state assembly
draw_box(ax, (0.0, 6.8), 1.3, 0.7, "PC=0\n(hard)", C_PC, fontsize=8)
draw_box(ax, (1.5, 6.8), 2.0, 0.7, "@0 code_embs\n(gradient ✓)", C_CODE, fontsize=8)
draw_box(ax, (3.7, 6.8), 2.0, 0.7, "@3 code_embs\n(detached ✗)", C_DETACH, fontsize=8)
draw_box(ax, (5.9, 6.8), 1.2, 0.7, "data\n(input)", C_DATA, fontsize=8)

# Arrows from code_params to input
draw_arrow(ax, (1.75, 9.0), (2.5, 7.5), color=C_CODE)
draw_arrow(ax, (5.25, 9.0), (4.7, 7.5), color=C_DETACH, ls="--")

# Model box
draw_box(ax, (1.0, 5.5), 5.5, 0.8, "Neural Executor  (frozen transformer)", C_MODEL, fontsize=10, bold=True)
draw_arrow(ax, (3.5, 6.8), (3.5, 6.3), color="black", lw=2)

# Output logits
draw_box(ax, (1.5, 4.2), 4.5, 0.7, "logits₀  (predictions over 256 tokens × 33 positions)", C_STATE, fontsize=8)
draw_arrow(ax, (3.75, 5.5), (3.75, 4.9), color="black", lw=2)

# --- Intermediate state construction ---
ax.text(10.5, 8.2, "Intermediate State Construction", fontsize=10, fontweight="bold",
        ha="center", color="#333")

draw_box(ax, (8.5, 6.8), 1.5, 0.7, "PC=3\n(hard-coded)", C_PC, fontsize=8)
draw_box(ax, (10.2, 6.8), 2.2, 0.7, "orig code embs\n(from input)", C_INPUT, fontsize=8)
draw_box(ax, (12.6, 6.8), 2.2, 0.7, "STE(logits₀)\ndata cells only", C_STATE, fontsize=8)

draw_arrow(ax, (6.0, 4.55), (12.8, 6.8), color="black", lw=1.5)
ax.text(10.5, 5.8, "argmax fwd\nsoftmax bwd", fontsize=7, ha="center",
        color="#666", style="italic")

# --- STEP 1 ---
ax.text(10.5, 4.5, "STEP 1  (PC=3, executes @3)", fontsize=11, fontweight="bold",
        ha="center", color="#333")

# Input state assembly for step 1
draw_box(ax, (7.5, 3.0), 1.3, 0.7, "PC=3\n(hard)", C_PC, fontsize=8)
draw_box(ax, (9.0, 3.0), 2.0, 0.7, "@0 code_embs\n(detached ✗)", C_DETACH, fontsize=8)
draw_box(ax, (11.2, 3.0), 2.0, 0.7, "@3 code_embs\n(gradient ✓)", C_CODE, fontsize=8)
draw_box(ax, (13.4, 3.0), 1.2, 0.7, "data\n(from STE)", C_STATE, fontsize=8)

# Arrows from intermediate state to step 1 input
draw_arrow(ax, (9.25, 6.8), (8.15, 3.7), color=C_PC, lw=1.5)
draw_arrow(ax, (11.3, 6.8), (10.0, 3.7), color=C_INPUT, lw=1.5)
draw_arrow(ax, (13.7, 6.8), (14.0, 3.7), color=C_STATE, lw=1.5)

# Arrows from code_params to step 1
draw_arrow(ax, (1.75, 9.0), (9.5, 3.7), color=C_DETACH, ls="--")
draw_arrow(ax, (5.25, 9.0), (12.0, 3.7), color=C_CODE)

# Model box for step 1
draw_box(ax, (8.0, 1.7), 5.5, 0.8, "Neural Executor  (frozen transformer)", C_MODEL, fontsize=10, bold=True)
draw_arrow(ax, (10.75, 3.0), (10.75, 2.5), color="black", lw=2)

# Final output
draw_box(ax, (8.5, 0.3), 4.5, 0.7, "logits₁  → loss = CE(logits₁[data], target)", C_LOSS, fontsize=9, bold=True)
draw_arrow(ax, (10.75, 1.7), (10.75, 1.0), color="black", lw=2)

# Legend
legend_items = [
    (C_CODE, "Learnable code (with gradient)"),
    (C_DETACH, "Detached code (no gradient through model)"),
    (C_PC, "Hard-coded PC"),
    (C_STATE, "STE intermediate state"),
    (C_MODEL, "Frozen neural executor"),
    (C_LOSS, "Loss (final only)"),
]
for idx, (color, label) in enumerate(legend_items):
    y = 1.5 - idx * 0.4
    rect = mpatches.FancyBboxPatch((-0.3, y - 0.12), 0.4, 0.24, boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.add_patch(rect)
    ax.text(0.3, y, label, fontsize=8, va="center")


# ============================================================
# BOTTOM: Backward pass (gradient flow)
# ============================================================
ax = axes[1]
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-0.5, 10.5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Backward Pass — Gradient Flow (k=2, loss-scope=final)", fontsize=14, fontweight="bold", pad=15)

# Loss at bottom
draw_box(ax, (5.5, 0.3), 5.0, 0.8, "∂L/∂logits₁  (from CE loss on final output)", C_LOSS, fontsize=9, bold=True)

# Step 1 model
draw_box(ax, (4.5, 2.0), 7.0, 0.9, "Backprop through Step 1 Model (frozen weights)", C_MODEL, fontsize=10, bold=True)
draw_arrow(ax, (8.0, 1.1), (8.0, 2.0), color=C_GRAD, lw=3)

# Gradient splits
ax.text(8.0, 3.3, "Gradients split to:", fontsize=9, fontweight="bold", ha="center")

# @3 gradient (direct)
draw_box(ax, (10.5, 4.0), 3.5, 0.8, "∂L/∂code_embs[@3]\n→ updates a₁, b₁, c₁", C_CODE, fontsize=8, bold=True)
draw_arrow(ax, (10.0, 2.9), (12.25, 4.0), color=C_GRAD, lw=3)
ax.text(11.5, 3.5, "DIRECT\n(strong)", fontsize=8, color=C_GRAD, fontweight="bold", ha="center")

# @0 gradient (blocked by detach)
draw_box(ax, (0.5, 4.0), 3.5, 0.8, "∂L/∂code_embs[@0]\n= 0 (detached at step 1)", C_DETACH, fontsize=8)
draw_arrow(ax, (5.5, 2.9), (2.25, 4.0), color=C_DETACH, lw=2, ls="--")
ax.text(3.0, 3.5, "BLOCKED\n(detached)", fontsize=8, color="#999", ha="center")

# Data cell gradient (the critical path)
draw_box(ax, (5.0, 4.0), 4.5, 0.8, "∂L/∂data_cells_intermediate\n(what data SHOULD have been)", C_STATE, fontsize=8, bold=True)
draw_arrow(ax, (8.0, 2.9), (7.25, 4.0), color=C_GRAD, lw=3)

# STE bottleneck
draw_box(ax, (4.5, 5.5), 5.5, 1.0, "STE Bottleneck\nfwd: argmax (discrete, no true gradient)\nbwd: softmax approx (noisy estimate)", "#FFCCCC", fontsize=8, bold=True)
draw_arrow(ax, (7.25, 4.8), (7.25, 5.5), color=C_GRAD, lw=3)

# Step 0 model
draw_box(ax, (4.5, 7.2), 7.0, 0.9, "Backprop through Step 0 Model (frozen weights)", C_MODEL, fontsize=10, bold=True)
draw_arrow(ax, (7.25, 6.5), (7.25, 7.2), color=C_GRAD, lw=2, ls="--")

# @0 gradient (indirect through data cells)
draw_box(ax, (1.0, 8.8), 4.5, 0.8, "∂L/∂code_embs[@0]\n→ updates a₀, b₀, c₀\n(INDIRECT, through STE)", C_CODE, fontsize=8, bold=True)
draw_arrow(ax, (5.5, 8.1), (3.25, 8.8), color=C_GRAD, lw=2, ls="--")

# The problem annotation
draw_box(ax, (8.5, 8.8), 5.5, 1.2,
         "THE PROBLEM:\n"
         "With random @0, intermediate data is garbage.\n"
         "STE grad through garbage → noise for @0.\n"
         "@0 can't improve → data stays garbage → stuck.",
         "#FFE0E0", fontsize=8, bold=True)

# Annotation arrows
ax.annotate("", xy=(8.5, 9.4), xytext=(5.5, 9.2),
            arrowprops=dict(arrowstyle="->", color="#CC0000", lw=2))

# Key insight box
draw_box(ax, (0.0, -0.3), 15.0, 0.5,
         "Key: @3 gets strong direct gradients.  @0 only gets indirect gradients through the STE bottleneck — this is why loss-scope=final fails from scratch.",
         "#FFF3CD", fontsize=9, bold=True)

plt.tight_layout()
plt.savefig("program_synthesis/k2_architecture_diagram.png", dpi=150, bbox_inches="tight")
plt.savefig("program_synthesis/k2_architecture_diagram.pdf", bbox_inches="tight")
print("Saved to program_synthesis/k2_architecture_diagram.png and .pdf")
