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