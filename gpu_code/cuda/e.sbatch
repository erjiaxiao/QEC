#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128
#SBATCH --gres=gpu:pascal:1
