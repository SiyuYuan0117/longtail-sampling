# Mini-batch Sampling Strategies for Long-Tailed Image Classification

Code, logs and figures for the paper *Mini-batch Sampling Strategies for
Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT*
(Siyu Yuan, University of Bristol).

The study compares four mini-batch samplers — uniform instance, class-balanced,
square-root and progressively balanced — under a controlled protocol on
CIFAR-100-LT with ResNet-32, at imbalance ratios rho in {10, 50, 100} and three
seeds each (36 runs in total). Within a seed, every strategy sees the identical
long-tailed subset and identical initial weights, so differences are
attributable to mini-batch composition alone.

## Headline result

Progressive sampling improves **tail-class** accuracy by 25% relative to the
uniform baseline at rho = 100 (13.5% vs 10.8%), consistently across all three
seeds. Its **overall** accuracy is *not* separable from uniform sampling at this
sample size. At rho = 100, class-balanced sampling degrades accuracy on every
class group, including the tail classes it is meant to help.

## Repository layout

```
CIFAR100LT_Experiments.py   All experiment code: dataset construction, the
                            four samplers, ResNet-32, training loop,
                            evaluation, table and figure generation.
CIFAR100LT_Run.ipynb        Notebook used to drive the runs.
results/                    Per-run JSON logs (36 files) + generated .tex tables.
figures/                    Generated PNG figures.
paper/                      LaTeX source of the paper and its PDF figures.
```

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
| `final_*_acc` | float | **values at the best-overall epoch, not epoch 200** |
| `final_per_class_acc` | list[100] | per-class accuracy at that epoch |

> **Note on `final_*` naming.** Despite the prefix, `final_overall_acc`,
> `final_head_acc`, `final_medium_acc` and `final_tail_acc` are recorded at the
> epoch that maximises `test_acc`, not at epoch 200. For genuine final-epoch
> values use the last element of the corresponding per-epoch list, e.g.
> `test_acc[-1]`. Appendix D of the paper reports both protocols.

## Reproducing

```bash
pip install -r requirements.txt
python CIFAR100LT_Experiments.py
```

CIFAR-100 is downloaded automatically into `./data` on first run (not tracked
here). A full sweep is 36 runs of 200 epochs. Seeds, cuDNN determinism and
per-worker DataLoader seeding are all fixed, so runs are reproducible on
identical hardware.

## Citation

```bibtex
@misc{yuan2026sampling,
  title  = {Mini-batch Sampling Strategies for Long-Tailed Image Classification:
            An Empirical Study on {CIFAR-100-LT}},
  author = {Yuan, Siyu},
  year   = {2026},
  eprint = {arXiv:2609.16365},
  primaryClass = {cs.LG}
}
```

## Licence

Code released under the MIT Licence (see `LICENSE`).
