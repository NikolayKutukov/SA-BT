You want your evaluation section to show **what** you measure, **how** you measure it, and **how** you aggregate/compare models. A concise structure that aligns with current recommendations is:

***

## 1. Define evaluation goals and metrics

State that you evaluate models along **three aspects**:

1. **Discrimination** – can the model rank patients by risk?
   - Use **C‑index**, but specify the variant:
     - Harrell’s C for baseline, and/or  
     - IPCW‑weighted C (e.g. Uno’s) to better handle censoring. [scikit-survival.readthedocs](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)

2. **Calibration / overall accuracy** – are predicted survival probabilities/time‑to‑event accurate?
   - **Time‑dependent Brier score** at clinically relevant times. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8484151/)
   - **Integrated Brier Score (IBS)** as a global summary over time. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12542389/)

3. **Time‑to‑event accuracy / interpretability (optional but modern)**
   - Include **MAE‑based metrics** such as MAE‑PO (pseudo‑observation–based MAE), which approximate the true MAE under censoring and have been argued to be more interpretable than pure ranking metrics. [proceedings.mlr](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf)

You can reference recent work that explicitly recommends moving beyond C‑index and combining discrimination and proper scoring rules. [arxiv](https://arxiv.org/html/2506.02075v1)

***

## 2. Describe the evaluation protocol

For each **scenario × sample size**:

1. **Data splitting / resampling**
   - Use **nested CV or repeated K‑fold CV**:
     - Inner loop: hyperparameter tuning.  
     - Outer loop: performance estimation.  
   - Clearly state K (e.g. 5), number of repeats, and that test folds are never used for tuning. [cran.r-project](https://cran.r-project.org/web/packages/survcompare/survcompare.pdf)

2. **Metric computation**
   - For each outer fold and replicate:
     - Compute C‑index (and variant), Brier score over a time grid, IBS, and (if used) MAE‑PO based on pseudo‑observations. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12396822/)
   - If you want time‑specific performance, specify a set of times (e.g. quartiles of event times or clinically relevant horizons). [scikit-survival.readthedocs](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)

3. **Aggregation**
   - Average metrics across outer folds and Monte Carlo repeats for each (model, n, scenario) combination.  
   - Report mean and variability (SD or confidence intervals). [arxiv](http://arxiv.org/pdf/2406.04098.pdf)

***

## 3. Summaries and visualizations

Use a few consistent views:

1. **Per‑scenario tables**
   - For each scenario, a table with rows = models, columns = metrics (C‑index, IBS, possibly MAE‑PO) at one fixed sample size.  
   - Include mean ± SD or CI. [journal.r-project](https://journal.r-project.org/articles/RJ-2023-009/)

2. **Heatmaps over sample size**
   - As you suggested: per scenario, heatmaps with X = sample size, Y = models, cell = metric (e.g. C‑index or IBS).  
   - This acts as a **learning curve visualization** and clearly shows sample‑size sensitivity. [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/sim.9931)

3. **Calibration plots (selected cases)**
   - For a subset of models and scenarios, show calibration (e.g. predicted vs observed survival at time t, or A‑calibration curves) to illustrate how models differ beyond ranking. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8484151/)

***

## 4. Statistical comparison

Explain how you formally compare methods:

- Use **ranks and average rank** across scenarios and sample sizes. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC9338425/)
- Optionally apply **Multiple Comparisons with the Best (MCB)** or similar procedures to test whether differences from the best model are statistically significant. [github](https://github.com/nliulab/Survival-Benchmark)
- Make clear that you do not over‑interpret tiny C‑index differences without considering IBS / MAE‑PO and uncertainty. [pure.au](https://pure.au.dk/portal/en/publications/stop-chasing-the-c-index-this-is-how-we-should-evaluate-our-survi/)

***

## 5. How to write it in your paper

Your evaluation section could be structured like:

1. **Evaluation metrics** (1–2 paragraphs): define C‑index variant(s), Brier score, IBS, and MAE‑PO, with short rationale and references. [proceedings.mlr](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf)
2. **Resampling and tuning protocol**: nested or repeated CV, metrics used in inner loop vs outer loop, aggregation. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12674930/)
3. **Reporting and visualization**: heatmaps over sample size, summary tables, calibration plots, and statistical tests/ranks. [academic.oup](https://academic.oup.com/bib/article/22/3/bbaa167/5895463)

This gives you a modern, multi‑metric evaluation aligned with recent “stop chasing the C‑index” recommendations, while still being compact and implementable for your benchmark. [arxiv](https://www.arxiv.org/abs/2506.02075)