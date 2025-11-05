# Cross-Layer Transcoders

This week, I wanted to explore some the concept of a cross-layer transcoder (CLT), which was described in one of the Anthropic interpretability research blog posts here: [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). The big goal of this network approach is to try and create a series of sparse feature sets for each layer that can recreate the original layer's outputs. This repo is forked from [etredal/openCLT](https://github.com/etredal/openCLT) and the CLT implementation is not my own minus just getting it working.

This network has some key pieces that I would like to point out:

- Training goal of each layer is to optimize for two types of loss: MSE loss for the layer reconstruction and sparsity loss which has a regularization term that acts as a downward pressure to keep features activating infrequently

- Reconstruction of layer outputs is made from ***decoders used for all layers prior to the layer being decoded*** (not sure how I feel about this, but this is the practice with the network leading to $\frac{L(L+1)}{2}$ decoders being trained in the model where $L$ is the number of layers in the base transformer)

- Despite having all these extra decoders, the outputs may not match up completely with the expected outputs of layers. To ensure that the attention maps still work when doing inference with the sparse features, error nodes are used to represent non-interpetable quantities that are needed to get back to the original outputs for layers

In my work, I just adjusted the codebase to work with `uv` instead of `poetry` and getting this running on one of the DGX nodes on ROSIE. In analyzing the CLT that was produced from the training (took about 4 hours on a single V100 GPU to train for a single epoch with the dataset in `examples/practice_run.py`). Looking at the attribution graphs that were created (`examples/run_example`), the goal is to show interpretable features that are linked together, but the model produced seemed to not really show what these features actually mean. To do this, I think there would have to be additional work of trying to find the most common locations of the features activating and using an LLM to label these features (at least that's what I have seen in other descriptions of feature creation from interpretability research).

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