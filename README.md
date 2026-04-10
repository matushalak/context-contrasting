# context-contrasting
Self-contained repository for the former `cc/circuit` and `cc/minimal` code.

## Layout
- `context_contrasting/circuit`: circuit-level contextual-contrasting models and experiments
- `context_contrasting/minimal`: minimal single-neuron contextual-contrasting model
- `context_contrasting/utils.py`: shared utility functions used by both model families
- `model_sketches`: static figures and sketches

## Setup
Create the environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate context-contrasting
python -m pip install -e .
```

After that, imports such as `from context_contrasting.circuit import Circuit` and `from context_contrasting.minimal.minimal import CCNeuron` should resolve from this repository alone.
