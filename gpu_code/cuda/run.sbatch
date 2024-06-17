#!/bin/bash
#SBATCH --partition=general
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128
#SBATCH --gres=gpu:1
#SBATCH -o ./data2/d9_q0_256_64_t2_r1
module load cuda/10.0
./execu -d 9 -t 2 -s 0 -q 0 -r 1 -1 256 -2 64
