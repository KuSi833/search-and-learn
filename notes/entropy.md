## Entropy, selection, and re-run strategy

### What exists

- The visualiser in `scripts/inference_visualiser.py` supports per-sample inspection, per-completion PRM summaries, and correctness via extracted answers. It can export subsets of incorrect items and uncertainty-ranked subsets using agreement, frequency entropy, PRM statistics, and grouped PRM mass. It can compare two runs via correctness transitions and report a fused accuracy that favours the re-run on the selected subset.

### Observed issue

- Re-running search on an uncertainty-selected subset often yields negligible or negative net gain in accuracy. This indicates leakage of true items into the subset and the presence of false items that are not recoverable by additional compute.

### Likely causes

- The selector mixes recoverable false items with always-false items that resist improvement, while also capturing fragile true items that flip under re-sampling.
- The fusion policy replaces original answers on the subset regardless of confidence, which can convert stable true items into false ones.

### Strategy

- Better selection: combine multiple uncertainty signals with a calibrated selector to limit true leakage while retaining recall for false items. Useful features include agreement, frequency entropy, PRM mean and standard deviation, PRM margin, top-mass within answer groups, the number of unique extracted answers, and the count of completions. Target a user-defined precision on false identification using simple calibration or conformal methods.
- Smarter re-runs: avoid always-false and always-true regions. Gate by thresholds on high agreement and large PRM margin to skip easy true items. Gate by very small PRM values across completions and diffuse mass to skip hopeless false items. Spend more tokens only on samples that show moderate agreement, middling margins, or disagreement across groups.
- Confidence-based fusion: keep the original answer unless the re-run exhibits strictly higher confidence according to a chosen score such as grouped top-mass or PRM margin. Optionally ensemble by weighted vote across the original and re-run predictions using PRM-normalised weights.
- Answer-aware verification: condition the re-run with a verify-and-correct prompt that shows the previous final answer and asks for a check before revising.

### Near-term goals

- Implement confidence-guided fusion that chooses between original and re-run based on a monotonic confidence score rather than default replacement.
- Add a light-weight selector that combines existing metrics to prioritise likely-false but recoverable cases while capping true leakage at a user-set level.
- Introduce stability filters to exclude always-false and always-true regions from re-runs, saving compute without harming accuracy.
- Evaluate coverage versus recall of false items and the effect on net accuracy under the new selection and fusion rules.
