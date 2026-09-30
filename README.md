# Context contrasting

Two studies, with separate entry points:

- **[thesis/](thesis/README.md)** — original models, population scatterplots, population perturbations, ablation visualizations, and the alpha sweep.
- **[paper_v2/](paper_v2/README.md)** — six independent mean-field models initialized from YAML, their robustness checks, and plots from two CSVs.

Run commands from this directory:

```bash
uv sync
uv run python -m paper_v2.simulate
uv run python -m paper_v2.plot
```

For plotting alone, the saved `paper_v2/ground_truth.csv` and `paper_v2/model.csv` are sufficient. Edit `paper_v2/configs/` to change model parameters or the experimental protocol.

The existing `context_contrasting/paper/` is preserved as a reference, including its local edits, checkpoints, and exploratory outputs. The old top-level `output/` and `tmp/` are also left as reference material. Active development belongs in `thesis/` or `paper_v2/`.

Original code outside `paper/` was moved to `thesis/`. A small import compatibility path in `context_contrasting/__init__.py` and a `data_analysis` symlink keep the old reference scripts usable. New code should import `thesis` or `paper_v2` directly.

Validation:

```bash
uv run python -m unittest discover -s tests -v
```
