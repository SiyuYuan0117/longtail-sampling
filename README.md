# Mini-batch Sampling Strategies for Long-Tailed Image Classification

[![arXiv](https://img.shields.io/badge/arXiv-2609.16365-b31b1b.svg)](https://arxiv.org/abs/2609.16365)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code, per-run logs and figures for the paper:

> **Mini-batch Sampling Strategies for Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT**
> Siyu Yuan, University of Bristol
> [arXiv:2609.16365](https://arxiv.org/abs/2609.16365) · [PDF](https://arxiv.org/pdf/2609.16365)

---

## What this is

Four mini-batch samplers are compared under a controlled protocol on CIFAR-100-LT
with ResNet-32, at imbalance ratios rho in {10, 50, 100} with three seeds each —
36 training runs in total.

| Sampler | Class sampling probability |
|---|---|
| **Uniform** (instance-balanced) | `P(k) = n_k / n` |
| **Class-balanced** | `P(k) = 1 / K` |
| **Square-root** | `P(k) = sqrt(n_k) / sum_j sqrt(n_j)` |
| **Progressive** | `P_t(k) = (1-lambda(t)) * n_k/n + lambda(t) * 1/K`, lambda rising 0 to 1 |

The samplers themselves are not new — they were assembled and contrasted by
[Kang et al. (ICLR 2020)](https://arxiv.org/abs/1910.09217). What this study adds is a
bias–variance treatment of the mini-batch gradient estimator that yields falsifiable
predictions, a controlled comparison in which *every strategy sees the identical
long-tailed subset and identical initial weights within a seed*, per-group training
dynamics rather than end-point numbers only, and a diagnosis of a failure mode that
aggregate results obscure.

## Headline findings

**1. Progressive sampling helps the tail, not the average.**

At rho = 100 it improves tail-class accuracy by 25% relative to the uniform baseline
(13.5% vs 10.8%). Because all strategies share seeds, runs can be paired — and the
paired differences separate a robust effect from noise:

| Metric | rho | seed 42 | seed 123 | seed 456 | Agreement |
|---|---|---|---|---|---|
| Overall | 100 | +0.61 | +0.52 | **−0.27** | 2/3 |
| Overall | 50 | +0.54 | +0.42 | **−0.85** | 2/3 |
| **Tail** | 100 | +2.13 | +3.23 | +2.81 | **3/3** |
| **Tail** | 50 | +4.00 | +2.48 | +0.63 | **3/3** |

*(progressive minus uniform, percentage points)*

The overall-accuracy lead is **not** robust; the tail-class gain is. The paper's
conclusions rest only on the latter.

**2. Class-balanced sampling backfires under severe imbalance.**

At rho = 100 it is worse than uniform on *every* class group — including the tail classes
it is designed to help (9.2% vs 10.8%). With only 5 samples per tail class, extreme
oversampling inflates the variance of the tail-class gradient contribution by roughly
two orders of magnitude. At the milder rho = 50, where the smallest classes hold 9
samples, the failure is confined to head and medium classes.

**3. Sampler choice only matters when imbalance is severe.**

The spread between the best and worst strategy is 1.6 points at rho = 10 and 6.0 at
rho = 100.

### Overall test accuracy (%)

| Strategy | rho = 10 | rho = 50 | rho = 100 |
|---|---|---|---|
| Uniform | 56.6 ± 0.1 | 44.1 ± 0.3 | 39.7 ± 0.4 |
| Class-Balanced | 56.0 ± 0.1 | 39.9 ± 0.2 | 34.0 ± 0.9 |
| Square-Root | 56.7 ± 0.1 | 43.0 ± 0.2 | 38.1 ± 0.2 |
| Progressive | **57.6 ± 0.2** | **44.1 ± 0.5** | **40.0 ± 0.6** |

### Per-group accuracy at rho = 100 (%)

| Strategy | Overall | Head | Medium | Tail |
|---|---|---|---|---|
| Uniform | 39.7 | **66.7** | 38.4 | 10.8 |
| Class-Balanced | 34.0 | 57.7 | 32.1 | 9.2 |
| Square-Root | 38.1 | 63.8 | 36.5 | 10.9 |
| Progressive | **40.0** | 64.9 | **38.6** | **13.5** |

Head = classes with more than 100 training samples, Medium = 20–100, Tail = 20 or fewer.
All numbers are means over seeds {42, 123, 456}; the reported spread is population sigma.

## Repository layout

```
CIFAR100LT_Experiments.py   All experiment code: CIFAR-100-LT construction, the four
                            samplers, ResNet-32, training loop, evaluation, and the
                            table/figure generation used in the paper.
CIFAR100LT_Run.ipynb        Notebook used to drive the runs.
results/                    36 per-run JSON logs + the generated .tex tables.
figures/                    Generated PNG figures.
paper/                      LaTeX source of the paper and its PDF figures.
```

## Reproducing

```bash
pip install -r requirements.txt
python CIFAR100LT_Experiments.py
```

CIFAR-100 downloads automatically into `./data` on first run (not tracked here).
A full sweep is 36 runs of 200 epochs. Python 3.10 / PyTorch 2.1.

Determinism is fixed throughout — Python, NumPy and PyTorch seeds, cuDNN
deterministic mode, and per-worker DataLoader seeding — so runs are reproducible on
identical hardware. Within a seed, the dataset subset and the initial weights are
shared across all four samplers, which is what makes the paired comparison valid.

### Training configuration

| Setting | Value |
|---|---|
| Optimiser | SGD, Nesterov momentum 0.9 |
| Weight decay | 2e-4 |
| Learning rate | 0.1, multiplied by 0.01 at epochs 160 and 180 |
| Warmup | 5 epochs, linear |
| Batch size | 128 |
| Epochs | 200 |
| Gradient clipping | max-norm 5.0 |

## Result log format

One JSON per run, named `results_<strategy>_rho<R>_seed<S>.json`:

| Key | Type | Meaning |
|---|---|---|
| `strategy` | str | `uniform`, `class_balanced`, `square_root`, `progressive` |
| `imbalance_ratio` | int | 10, 50 or 100 |
| `seed` | int | 42, 123 or 456 |
| `class_counts` | list[100] | training samples per class |
| `config` | dict | full hyperparameter record |
| `train_loss`, `train_acc` | list[200] | per-epoch training curves |
| `test_acc`, `test_loss` | list[200] | per-epoch test curves |
| `head_acc`, `medium_acc`, `tail_acc` | list[200] | per-epoch group accuracy |
| `lr_history` | list[200] | per-epoch learning rate |
| `best_acc` | float | max over `test_acc` |
| `final_*_acc` | float | **values at the best epoch, not epoch 200** — see below |
| `final_per_class_acc` | list[100] | per-class accuracy at that epoch |

> **Naming caveat.** Despite the prefix, `final_overall_acc`, `final_head_acc`,
> `final_medium_acc` and `final_tail_acc` are recorded at the epoch maximising
> `test_acc`, **not** at epoch 200. For genuine final-epoch values use the last
> element of the corresponding per-epoch list, e.g. `test_acc[-1]`.
> Appendix D of the paper reports both protocols and shows the difference is
> 0.06–0.30 percentage points, with the tail-class finding holding under either.

Minimal example:

```python
import json, glob

runs = {}
for f in glob.glob("results/results_*.json"):
    d = json.load(open(f))
    runs[(d["strategy"], d["imbalance_ratio"], d["seed"])] = d

# tail accuracy at the best epoch, progressive vs uniform, rho = 100
for s in (42, 123, 456):
    p, u = runs[("progressive", 100, s)], runs[("uniform", 100, s)]
    print(s, round(p["final_tail_acc"] - u["final_tail_acc"], 2))
```

## Citation

```bibtex
@misc{yuan2026minibatch,
  title         = {Mini-batch Sampling Strategies for Long-Tailed Image Classification:
                   An Empirical Study on {CIFAR-100-LT}},
  author        = {Yuan, Siyu},
  year          = {2026},
  eprint        = {2609.16365},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2609.16365},
  url           = {https://arxiv.org/abs/2609.16365}
}
```

## Acknowledgements

This work began as an undergraduate individual project in the School of Mathematics,
University of Bristol, supervised by Dennis Prangle.

## Licence

MIT — see [LICENSE](LICENSE).
