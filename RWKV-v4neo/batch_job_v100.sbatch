#!/bin/bash

#SBATCH -JTrain_36M_V100_ctx512_lr1e-3                       # Job name
#SBATCH -N1 -t8:00:00 --gres=gpu:2 -C V100-32GB --ntasks-per-node=2 --mail-type=BEGIN,END,FAIL --mail-user=murilogustineli@gatech.edu
#SBATCH -oReport-%j.out                         # Combined output and error messages file

cd ~/scratch/scalable-RWKV/RWKV-v4neo/         # Change to working directory
nvidia-smi

module load anaconda3                           # Load module dependencies
conda activate rwkv_4neo
python train.py --load_model "" --wandb "" --proj_dir "out_36M_V100_ctx512_lr1e-3" \
--data_file "data/TinyStories-train" --data_type "binidx" --vocab_size 50277 \
--ctx_len 512 --epoch_steps 66200 --epoch_count 4 --epoch_begin 0 --epoch_save 1 \
--micro_bsz 16 --n_layer 12 --n_embd 256 --pre_ffn 0 --head_qk 0 \
--lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision fp16 --strategy ddp_find_unused_parameters_false \
--grad_cp 0 --dropout=0.001 --weight_decay=0.001
