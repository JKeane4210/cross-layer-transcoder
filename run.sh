#!/bin/bash

# Job metadata
#SBATCH --job-name='cross-layer-transcoder'
#SBATCH --output="%j.out"

# Hardware specs
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=8024

# Run time
## time format: <days>-<hours>:<minutes>
#SBATCH --time=1-0:0

# Email notification (uncomment and set email address to enable)
#SBATCH --mail-type=END
#SBATCH --mail-user="keanej@msoe.edu"

# command
uv run ./examples/practice_run.py