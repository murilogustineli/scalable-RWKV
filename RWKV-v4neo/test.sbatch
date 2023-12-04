#!/bin/bash

#SBATCH -Jtest_w_cpu                     # Job name
#SBATCH -N1 --ntasks-per-node=2     # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=8G                        # Memory per core
#SBATCH -t1:00:00                               # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=rbrock8@gatech.edu        # E-mail address for notifications
cd ~/scratch/scalable-RWKV/RWKV-v4neo/         # Change to working directory
nvidia-smi

module load anaconda3                           # Load module dependencies
conda activate rwkv_4neo

echo hello

ls

exit