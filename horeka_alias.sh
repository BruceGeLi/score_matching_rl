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

## Walker walk
alias walker_walk='runexp ./cw_configs/_walker_walk/horeka.yaml   -o -s'
