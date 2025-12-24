#!/bin/bash

#SBATCH -p yuanlab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 1800G
#SBATCH -c 120
#SBATCH -J TED
#SBATCH -o /storage/yuanfajieLab/yuanfajie/fengyuan/MegaDLMs/output/SlurmLogs/%j.log
#SBATCH --gres=gpu:8

bash $1 