#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --output=slurm-output.out
#SBATCH --error=slurm-error.err

module load python/3.9.0
module load py-pandas
module load py-pytorch/1.12.0_py39
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

python3 /home/groups/risheej/tom/dbr/
