# GPT-RWKV
An evaluation of RWKV scalability using limited computing resources based on computational efficiency and memory usage on different benchmark tasks.


## Connecting to Georgia Tech VPN
### Connecting to vpn.gatech.edu
With Mac/Linux this should work:

`sudo openconnect --protocol=gp -u <GaTech user name> vpn.gatech.edu`

- Follow password prompts (first sudo, then gatech, then Duo 2FA)
- Select "DC Gateway"
Then just minimize that terminal.

### SSH to PACE-ICE
In terminal:

`ssh <username>@login-ice.pace.gatech.edu`

You'll probably want to use terminal to set up Conda below.

### Conda Symlink Set Up
1. Check you have a ~/.conda directory by running `file ~/.conda`
2. If `file ~/.conda` reports "No such file or directory" then create one in scratch directory `mkdir ~/scratch/.conda` and create a symlink `ln -s ~/scratch/.conda ~/.conda`
3. If `file ~/.conda` reports that `~/.conda` is a directory, then move that directory to scratch directory: `mv ~/.conda ~/scratch` and create a symlink `'ln -s ~/scratch/.conda ~/.conda`
4. If `file ~/.conda` reports it is a symlink then make sure it is pointing where you want it to.

### Start Conda
From within the terminal after you've ssh'd in: `module load anaconda3`

_Every time you log in to the head node or a compute node, you'll need to do the above and then load your rwkv_4neo environment which is set up below._


## Environment Setup
### Conda Environment Setup
You can't specify extra-index-url from within environment file so set it up manually

1. Create and activate the conda env
```
conda create -y --name rwkv_4neo python=3.10

conda activate rwkv_4neo
```

2. Install cuda toolkits
```
conda install -y -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7 
```

3. Setting up pytorch 1.13.1 with cuda 1.17 specifically
```
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

python -m pip install deepspeed==0.7.0 pytorch-lightning==1.9.5`

python -m pip install ninja wandb transformers`
```

You can double check your symlink is working properly at this point with this command:
```
pace-quota
```


## PACE-ICE Use
### Useful Commands
```
$ pace-quota    #checks your storage for ~/ and ~/scratch
$ scontrol show nodes | grep -B 10 -A10 "State=IDLE"    #shows idle nodes
# In the above command, you can also do something like "A100" with grep
$ scontrol show nodes     #shows all nodes
```


### Connecting to Compute
You should be able to immediately connect to idle nodes, which can be done with the following:

```
> salloc -N1 --mem-per-gpu=16G -t0:15:00 --gres=gpu:V100:1 --ntasks-per-node=1

-N1    #allocates one node (this will probably always be just 1)
--mem-per-gpu    # how much RAM is reserved along each GPU (RAM, not VRAM!)
-t    # how much time until compute node shuts down automatically hh:mm:ss
-gres   # specify which gpu you want to allocate
-ntasks-per-node    # should match how many gpus you are trying to allocate
--mail-type=BEGIN,END,FAIL     # Mail preferences
--mail-user=gburdell3@gatech.edu

```

Here is the different ways you can specify which gpus to reserve

format --gres=gpu: gpu_type:num_gpus

- Nvidia Tesla V100
    - `--gres=gpu:V100:1` or `-G V100:1` for any V100
    - `--gres=gpu:1 -C V100-16GB` or `-G1 -C V100-16GB` for a V100 with 16 GB of memory
    - `--gres=gpu:1 -C V100-32GB` or `-G1 -C V100-32GB` for a V100 with 32 GB of memory
    - maximum 4 V100 per node
- Nvidia Quadro Pro RTX6000 (note underscore in some syntax)
    - `--gres=gpu:RTX_6000:1` or `-G RTX_6000:1` or `--gres=gpu:1 -C RTX6000` `-G 1 -C RTX6000`
    - maximum 4 RTX6000 per node
- Nvidia A40
    - `--gres=gpu:A40:1` or `-G A40:1` or `--gres=gpu:1 -C A40` or `-G 1 -C A40`
    - maximum 2 A40 per node with AMD CPUs
- Nvidia A100
    - `--gres=gpu:A100:1` or `-G A100:1` for any A100
    - `--gres=gpu:1 -C A100-40GB` or `-G 1 -C A100-40GB` for an A100 with 40 GB of memory
    - `--gres=gpu:1 -C A100-80GB` or `-G 1 -C A100-80GB` for an A100 with 80 GB of memory
    - maximum 2 A100 per node with AMD CPUs

__TO CONNECT TO 4 RTX_6000s (put in your own email):__
```
salloc -N1 --mem-per-gpu=16G -t5:00:00 --gres=gpu:RTX_6000:4 --ntasks-per-node=4 --mail-type=BEGIN,END,FAIL --mail-user=<gatech_user_name>@gatech.edu
```

## TRAINING NOTES
You can submit batch jobs by doing:
`sbatch batch_job.sbatch`

There are two `.sbatch` files in the repo (already in main). One is set up for V100 and the other is set up for A100. The V100 has VRAM specified. Both assume 2 GPUS are being allocated for the job.

You’ll need to configure the following parameters to your desired values:
- `n_layer`: _default_=`6`, _type_=`int`
- `n_embd`: _default_=`512`, _type_=`int`
- `ctx_len`: _default_=`1024`, _type_=`int`
- `proj_dir`: _default_=`"out"`, _type_=`str`

__IMPORTANT__.

Format `proj_dir` as `"out_modelsize_gpu_ctxSize_lrSize"`.
- Example: `"out_92M_V100_ctx1024_lr6e-4"`

**YOU NEED TO REMEMBER TO ADD YOUR `.out` FILE TO YOUR `out` FOLDER AFTER TRAINING.** The `"Report-######"` file. This has important information that will be used to make learning curves, track time, etc.

### Helpful Commands
For helpful commands on batch jobs, visit http://docs.pace.gatech.edu/gettingStarted/commands/.

To see your queued jobs:
`squeue -u <user-name>`

To see your active jobs in the browser:
1. Access https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard while connected to the VPN.
2. Click on "Jobs" at the top of the page, then "Active jobs"

### Check node once connected
