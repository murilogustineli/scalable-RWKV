#!/etc/sh

#SBATCH -JTrain_92M                       # Job name
#SBATCH -N1 --gres=gpu:A100:2 --ntasks-per-node=2     # Number of nodes and cores per node required
#SBATCH --mem-per-gpu=8G                        # Memory per core
#SBATCH -t8:00:00                               # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=rbrock8@gatech.edu        # E-mail address for notifications
cd ~/scratch/scalable-RWKV/RWKV-v4neo/         # Change to working directory
nvidia-smi

module load anaconda3                           # Load module dependencies
conda activate rwkv_4neo
srun python train.py --load_model "" --wandb "" --proj_dir "out_92M" \
--data_file "data/TinyStories-train" --data_type "binidx" --vocab_size 50277 \
--ctx_len 1024 --epoch_steps 66200 --epoch_count 5 --epoch_begin 0 --epoch_save 1 \
--micro_bsz 16 --n_layer 12 --n_embd 512 --pre_ffn 0 --head_qk 0 \
--lr_init 6e-4 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 2 --precision bf16 --strategy ddp_find_unused_parameters_false \
--grad_cp 0 --dropout=0.001 --weight_decay=0.001
