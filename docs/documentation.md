# Tiny Recursive Model (TRM) – Technical Documentation

## Overview

The **Tiny Recursive Model (TRM)** is a compact, iterative neural architecture designed to emulate a _reasoning-style_ refinement process for **Sudoku solving**. Rather than relying on deep transformer stacks, TRM uses explicit **inner–outer iterative loops** to repeatedly refine latent hypotheses before committing to an output.

Formally, the model learns a function

$$
f_\theta : \mathbb{R}^{810} \rightarrow \mathbb{R}^{81 \times 9}
$$

mapping a flattened one-hot Sudoku encoding to categorical logits over digits $1..9$ for each of the 81 cells.

This project is intentionally lightweight and pedagogical. The goal is clarity of reasoning dynamics, not state-of-the-art performance.

## Project Structure

- `tiny-recursive-model/tiny-recursive-model.ipynb` - main notebook (model definition, dataset, trainer, training & inference utilities).
- `data/sudoku.csv` - expected dataset (CSV with quizzes and solutions columns).
- `docs/documentation.md` - this file.
- `checkpoints/` - directory where checkpoint files are saved by default.
- `main.py` - optional script for running inference on new puzzles.

Use these names when you run or inspect the project. The notebook contains the runnable code used for training and inference.

## Model Architecture and Reasoning Flow

### Input Encoding

Each Sudoku puzzle is encoded as a tensor:

- 81 cells
- 10 channels per cell
  - channel 0 → empty
  - channels 1–9 → digits 1–9

Resulting in a flattened input vector:

$$
x \in \mathbb{R}^{81 \times 10} \equiv \mathbb{R}^{810}
$$

### Input Projection

The input is linearly projected into a latent space:

$$
x_{\text{emb}} = W_{\text{in}} x + b_{\text{in}}, \quad
x_{\text{emb}} \in \mathbb{R}^{1 \times d}
$$

where $d = \texttt{hiddenDim}$ (default $512$).

### Internal States

The model maintains two persistent internal states:

- **Latent hypothesis state**
  $$
  z \in \mathbb{R}^{1 \times d}
  $$
- **Output refinement state**
  $$
  y \in \mathbb{R}^{1 \times d}
  $$

Initialisation:

$$
z_0 = 0,\quad y_0 = 0
$$

### Iterative Reasoning Loops

TRM performs **nested recursion**:

#### Inner Loop (Latent Refinement)

For $l = 1 \dots L_{\text{cycles}}$:

$$
z_{l+1} = z_l + \alpha \cdot \mathcal{R}_\phi(x_{\text{emb}} + y + z_l)
$$

Where:

- $\mathcal{R}_\phi$ is a stack of residual MLP blocks
- $\alpha$ is a learnable scalar **latent gate**
- Residual blocks are _pure transforms_, gating is applied externally

This loop represents _hypothesis refinement_.

#### Outer Loop (Hypothesis Integration)

For $h = 1 \dots H_{\text{cycles}}$:

1. Run the full inner loop to convergence, producing $z^*$
2. Update output state:

$$
y_{h+1} = y_h + \beta \cdot \mathcal{O}_\psi(y_h + z^*)
$$

Where:

- $\mathcal{O}_\psi$ is a small output MLP stack
- $\beta$ is a learnable scalar **output gate**

This loop represents _committing refined hypotheses into the output belief_.

### Final Projection

After $H_{\text{cycles}}$ outer iterations:

$$
\hat{o} = W_{\text{out}} y + b_{\text{out}}
$$

with:

$$
\hat{o} \in \mathbb{R}^{81 \times 9}
$$

Each row corresponds to logits over digits $1..9$ for a single cell.

Prediction:

$$
\hat{d}_{i} = \arg\max_{k \in \{1..9\}} \hat{o}_{i,k}
$$

## Data and Dataset Format

The dataset is a CSV with two columns:

- `quizzes`: string of length 81, digits `0–9`
- `solutions`: string of length 81, digits `1–9`

Example:

quizzes = “003020600900305001…”
solutions = “483921657967345821…”

### Encoding

For each cell $c$:

- If $c = 0$, set one-hot index 0
- If $c = k$, set one-hot index $k$

Targets are converted as:

$$
y_{\text{target}} = \text{digit} - 1 \in \{0..8\}
$$

to satisfy `CrossEntropyLoss`.

## Configuration (TRMConfig)

Key hyperparameters:

| Parameter      | Meaning                           |
| -------------- | --------------------------------- |
| `input_dim`    | $81 \times 10 = 810$              |
| `hidden_dim`   | Latent width $d$                  |
| `output_dim`   | $81 \times 9 = 729$               |
| `L_layers`     | Residual blocks per latent update |
| `L_cycles`     | Inner refinement iterations       |
| `H_cycles`     | Outer integration iterations      |
| `dropout`      | MLP dropout                       |
| `lr`           | Learning rate                     |
| `weight_decay` | AdamW regularisation              |
| `batch_size`   | Training batch size               |
| `epochs`       | Training epochs                   |

All parameters live in a `TRMConfig` dataclass.

## Training Objective

The model is trained with **cell-wise categorical cross-entropy**:

$$
\mathcal{L}
= \frac{1}{81N}
\sum_{n=1}^{N}
\sum_{i=1}^{81}
\text{CE}\left(
\hat{o}_{n,i},\;
y_{n,i}
\right)
$$

Implementation detail:

- Output reshaped to `(batch × 81, 9)`
- Targets reshaped to `(batch × 81)`

Optimisation:

- AdamW
- Gradient clipping at $\|g\|_2 \le 1$
- Cosine annealing LR schedule

## Inference

Given a puzzle $q \in \{0..9\}^{81}$:

1. Encode to one-hot $x \in \mathbb{R}^{810}$
2. Forward pass through TRM
3. Reshape output to $(81, 9)$
4. Apply argmax and map $0..8 \rightarrow 1..9$

Note:

- The model predicts _all_ cells
- Preserve givens manually if desired

## Saving and Deployment

### Checkpoints

Saved checkpoints include:

- `model_state_dict`
- `optimizer_state_dict`
- `config`
- training metrics

### Production Artifact

A minimal file:

`trm_sudoku_production.pt`

containing only:

- model weights
- config

Reload with:

$$
\theta \leftarrow \text{loadStateDict}(\cdot)
$$

and switch to eval mode.

## Common Failure Modes

1. **Divergence**

   - Reduce $L_{\text{cycles}}$, $H_{\text{cycles}}$
   - Initialise $\alpha, \beta \approx 0$

2. **Slow Training**

   - Large recursion depth implies $O(L \cdot H)$ compute
   - Reduce `hidden_dim` or batch size

3. **Bad Accuracy**

   - Dataset corruption
   - Excessive gating early in training

4. **OOM**
   - Latent recursion is memory-expensive
   - Gradient checkpointing could help

## Extensions and Research Directions

- Replace scalar gates $\alpha, \beta$ with vectors $\in \mathbb{R}^d$
- Introduce attention across the 81-cell dimension
- Curriculum learning by puzzle difficulty
- Constraint-aware losses (row/column/subgrid penalties)
- Quantisation and pruning for edge deployment

## Credits

This project demonstrates that _explicit iterative refinement_ can substitute depth, at least for structured reasoning tasks like Sudoku. It is intentionally minimal, interpretable, and hackable.

> _**Note**: Some of the variable names are written in camel case inside the LaTeX blocks due to limitations with the markdown parser used._
