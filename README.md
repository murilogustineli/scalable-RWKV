# GPT-RWKV
An evaluation of RWKV scalability using limited computing resources based on computational efficiency and memory usage on different benchmark tasks.


## Environment Setup
### Connect to vpn.gatech.edu

With Mac/Linux this should work:
`sudo openconnect --protocol=gp -u <GaTech user name> vpn.gatech.edu`
- follow password prompts (first sudo, then gatech, then Duo 2FA)
- select "DC Gateway"
Then just minimize that terminal

### SSH to PACE-ICE
- In terminal: `ssh <username>@login-ice.pace.gatech.edu`
- You'll probably want to use terminal to set up Conda below

### Conda Symlink Set Up

1) Check you have a ~/.conda directory
2) If `file ~/.conda` reports "No such file or directory" then create one in scratch directory `mkdir ~/scratch/.conda` and create a symlink `ln -s ~/scratch/.conda ~/.conda`
3) If `file ~/.conda` reports that `~/.conda` is a directory, then move that directory to scratch directory: `mv ~/.conda ~/scratch` and create a symlink `'ln -s ~/scratch/.conda ~/.conda`
4) If `file ~/.conda` reports it is a symlink then make sure it is pointing where you want it to 

### Start Conda
From within the terminal after you've ssh'd in: `module load anaconda3`
_Every time you log in to the head node or a compute node, you'll need to do the above and then load you rwkv_4neo environment which is set up below_

### Conda Env. Set Up
You can't specify extra-index-url from within environment file so set it up manually
```
# Create and activate the conda env
conda create -y --name rwkv_4neo python=3.10
conda activate rwkv_4neo

# Install cuda toolkits
conda install -y -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7 

# Setting up pytorch 1.13.1 with cuda 1.17 specifically
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install deepspeed==0.7.0 pytorch-lightning==1.9.5
python -m pip install ninja wandb transformers
```

You can double check your symlink is working properly at this point with this command
```
pace-quota
```
