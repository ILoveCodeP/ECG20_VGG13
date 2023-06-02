#!/bin/bash
#SBATCH -J VGGECG       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 1          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04  # apply for node04
#SBATCH -o VggEcgTest.out

params=(
  # "--num_filter_block=16"
  # "--num_filter_block=32"
  # "--num_filter_block=64"
  # "--num_filter_block=128"
  "--num_filter_block=64 --nb_cnn=6"
)

for param in "${params[@]}"; do
  python testonlymix.py $param
  echo "Script completed: $param"
done
