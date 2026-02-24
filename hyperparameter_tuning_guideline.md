Here’s a concise, concrete tuning plan that fits your Ryzen 5 9600X + RTX 5070 Ti and the experimental design we discussed.

***

## 1. Overall strategy

- **Nested 5×5 CV per (scenario, n):**  
  - Inner 5‑fold: hyperparameter search.  
  - Outer 5‑fold: unbiased evaluation.  
- **Search method:** random search with fixed **number of configs per model–dataset**, not open‑ended time.  
- **Early stopping:** for all deep models, with a patience threshold on validation loss/IBS.

This is compute‑bounded, fair across models, and matches modern recommendations for nested CV + constrained HPO. [cran.r-project](https://cran.r-project.org/web/packages/survcompare/survcompare.pdf)

***

## 2. Budgets per model–dataset

Per **(scenario, sample size)**:

- **Classical models (CPU)**
  - Cox / penalized Cox, RSF, GBM (if used).  
  - **Configs:** ~30 per model.  
  - **Estimate:** ≤2–3 minutes per config (5‑fold CV) → **≤1–1.5 CPU‑hours per model–dataset**, easily parallelizable across 12 threads.

- **Deep models (GPU)**
  - DeepSurv, DeepHit, transformer.  
  - **Configs:** ~15 per model (10 for transformer if heavy).  
  - **Cap:** ≤15 minutes per config with early stopping → **≈3.75 GPU‑hours for 15 configs**.  
  - That’s manageable if you run 1–2 scenarios per night.

You can slightly reduce configs (e.g. 20 classical / 10 deep) if runtime is tight.

***

## 3. Search spaces (minimal but effective)

Use **small, sensible ranges** rather than huge grids.

### Cox / penalized Cox (CPU)

- Penalty type: ridge or elastic‑net (α ∈ {0, 0.5, 1}).  
- λ: log‑spaced, e.g. 20 values from 1e‑4 to 10.  
- Tune on inner CV **C‑index or IBS**.

### Random Survival Forest (CPU)

- n_estimators: {200, 500, 1000}.  
- max_features: {√p, p/3, p/2}.  
- min_samples_leaf: {5, 10, 20}.  
- max_depth: {None, 10, 20}.  
Randomly sample 30 combos from this grid.

### DeepSurv (GPU)

- Layers: {2, 3}.  
- Hidden units: {64, 128, 256}.  
- Dropout: {0.0, 0.2, 0.5}.  
- Learning rate: {1e‑4, 3e‑4, 1e‑3}.  
- Weight decay: {0, 1e‑4, 1e‑3}.  
Sample 15 configs; early stopping with patience, say 10 epochs.

### DeepHit (GPU)

- Similar LR/weight decay grid.  
- Shared/branch layers: {1–2}, units {64, 128}.  
- Dropout {0, 0.3, 0.5}.  
- Loss trade‑off hyperparams in a small grid (e.g., 3–4 values).

### Transformer (GPU)

Keep it **small**:

- Layers: {2, 4}.  
- Heads: {4, 8}.  
- Hidden dim: {128, 256}.  
- Dropout: {0.1, 0.3}.  
- LR: {5e‑5, 1e‑4, 3e‑4}.  
Sample 10 configs with strong early stopping and a relatively small max epoch count.

***

## 4. Implementing nested CV with tuning

For each **scenario + n**:

1. Generate / select dataset.  
2. Outer 5‑fold split.  
3. For each outer fold:
   - Inner 5‑fold CV on training part:
     - Run random search (fixed config count).  
     - For each config, train model with early stopping, compute inner‑CV metric.  
     - Select best config by chosen metric (e.g., IPCW C‑index).  
   - Re‑fit model on full outer‑training data with best hyperparameters.  
   - Evaluate on outer test fold (C‑index, IBS, etc.).  
4. Aggregate outer‑fold metrics across folds and MC replicates.

This **reuses the same protocol** across all models, which is crucial for fairness. [arxiv](http://arxiv.org/pdf/2406.04098.pdf)

***

## 5. Practical tips for your machine

- **CPU parallelization:**  
  - Use joblib / multiprocessing for inner‑CV folds of classical models; your 12 threads can handle multiple configs at once.  
- **GPU scheduling:**  
  - Run deep models sequentially per scenario; let them run overnight.  
  - Limit dataloader workers (e.g., 2–4) so CPU isn’t a bottleneck.

- **Pilot run:**  
  - Do one small pilot for each model on one scenario to measure actual time per config; then adjust config counts (e.g., 30→20 or 15→10) if needed.

***

## 6. How to write this in your Methods

You can describe it along these lines:

- “We used nested 5×5‑fold cross‑validation. Hyperparameters were tuned in the inner folds using random search with at most 30 configurations for classical models and 15 configurations for deep models, subject to a per‑configuration wall‑clock limit of 15 minutes on our RTX 5070 Ti GPU and Ryzen 5 9600X CPU.”  
- “Search spaces were restricted to a small set of widely used hyperparameters per model class (e.g., penalty strength for Cox, tree depth and mtry for RSF, network width/depth, dropout, and learning rate for deep models).”

This gives you a tuning setup that is:

- Statistically sound (nested CV),  
- Fair across models (fixed config counts),  
- Realistic for your hardware and the scale of your benchmark.