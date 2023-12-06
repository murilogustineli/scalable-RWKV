#!/bin/bash

#SBATCH -JTrain_753M_A100_4e-4                   # Job name
#SBATCH -N1 -t8:00:00 --gres=gpu:A100:2 --ntasks-per-node=2 --mail-type=BEGIN,END,FAIL --mail-user=rbrock8@gatech.edu
#SBATCH -oReport-%j.out                         # Combined output and error messages file
cd ~/scratch/scalable-RWKV/RWKV-v4neo/         # Change to working directory
nvidia-smi

module load anaconda3                           # Load module dependencies
conda activate rwkv_4neo
python train.py --load_model "" --wandb "" --proj_dir "out_753M_A100_4e-4 " \
--data_file "data/TinyStories-train" --data_type "binidx" --vocab_size 50277 \
--ctx_len 512 --epoch_steps 66200 --epoch_count 2 --epoch_begin 0 --epoch_save 1 \
--micro_bsz 16 --n_layer 18 --n_embd 1536 --pre_ffn 0 --head_qk 0 \
--lr_init 4e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 2 --precision bf16 --strategy ddp_find_unused_parameters_false \
--grad_cp 0 --dropout=0.001 --weight_decay=0.001 --random_seed=-1 > 753M_A100_4e-4.out
