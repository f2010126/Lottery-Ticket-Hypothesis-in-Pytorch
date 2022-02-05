#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-dsengupt
#SBATCH --gres=gpu:1
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/open_lth.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/open_lth.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J FB_OG
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL

cd $(ws_find lth_ws)
#python3 -m venv lth_env
source lth_env/bin/activate

pip list
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

cd OtherLTHFrameworks/Lottery-Ticket-Hypothesis-in-Pytorch
python3 -m main --prune_type=lt --arch_type=simsiam_resnet18 --trial lth_sim --dataset=cifar10 --prune_percent=20 --lr 1.2e-3 --prune_iterations=10 --end_iter=30
deactivate
