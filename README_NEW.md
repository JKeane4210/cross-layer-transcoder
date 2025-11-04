# Cross-Layer Transcoders

## Converting a `poetry` Project to `uv`

```
uvx pdm import pyproject.toml -v
```

In addition to this, there were a couple things that copilot cleaned up:

- dependency syntax (`poetry` has additional parentheses whereas `uv` doesn't have these)

- removed unnecessary reference to a workspace for the `src` directory

## Training Batch File for ROSIE

```
sbatch run.sh
```

Interestingly, this task is large enough where I can compare the estimated durations of training across different models.

- ROSIE: Tesla T4 (~10 hours)

- ROSIE: V100-SXM2 (~4 hours)

- rosita: RTX 4090 (~2 hours)

Additionally, for the Tesla T4, there was not enough memory on the GPU for this model, which required me to use the V100s at the minimum.