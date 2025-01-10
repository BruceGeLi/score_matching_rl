alias CD='cd'
alias LS='ls'
alias GIT='git'

# Slurm watch alias
alias wa='watch -n 20 squeue'
alias was='watch -n 20 squeue --start'
alias keep='watch -n 600 squeue'
alias sfree='sinfo_t_idle'
alias sgpu='salloc -p accelerated -n 1 -t 120 --mem=200000 --gres=gpu:1 --account=hk-project-p0022232'
alias sgpu_dev='salloc -p dev_accelerated -n 1 -t 60 --mem=200000 --gres=gpu:1 --account=hk-project-p0022232'
alias scpu='salloc -p cpuonly -n 1 -t 120 --mem=200000 --account=hk-project-p0022232'
alias scpu_dev='salloc -p dev_cpuonly -n 1 -t 120 --mem=200000 --account=hk-project-p0022232'


# cd alias
alias cdmprl='cd ~/projects/onur/score_matching_rl/examples/states'

# Git alias
alias gp='cdmprl && git pull'

# Env alias
alias vb='cd ~/ && vim .bashrc'
alias ss='cd ~/ && source .bashrc && conda activate onur_icml'

# Exp
alias runexp='cdmprl && python train_score_matching_online.py'

#################################  DMC #################################
## Walker walk
alias walker_walk='runexp ./cw_configs/_walker_walk/horeka.yaml   -o -s'

## Dog run
alias dog_run='runexp ./cw_configs/dog_run/horeka.yaml   -o -s'
alias dog_run_horeka_local='runexp ./cw_configs/dog_run/local.yaml   -o --nocodecopy'

## Dog walk
alias dog_walk='runexp ./cw_configs/dog_walk/horeka.yaml   -o -s'

## Dog stand
alias dog_stand='runexp ./cw_configs/dog_stand/horeka.yaml   -o -s'

## Dog trot
alias dog_trot='runexp ./cw_configs/dog_trot/horeka.yaml   -o -s'

## Humanoid run
alias humanoid_run='runexp ./cw_configs/humanoid_run/horeka.yaml   -o -s'

## Humanoid stand
alias humanoid_stand='runexp ./cw_configs/humanoid_stand/horeka.yaml   -o -s'

## Humanoid walk
alias humanoid_walk='runexp ./cw_configs/humanoid_walk/horeka.yaml   -o -s'


## Run all jobs together with 1 second delay in between
alias run_all_dmc='dog_run && sleep 1 && dog_walk && sleep 1 && dog_stand && sleep 1 && dog_trot && sleep 1 && humanoid_run && sleep 1 && humanoid_stand && sleep 1 && humanoid_walk && sleep 1'
