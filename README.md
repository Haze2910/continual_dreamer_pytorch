# Continual Dreamer Pytorch

This is a pytorch implementation of the Continual Dreamer model (https://arxiv.org/abs/2211.15944).


## How to use
1. Install the required dependencies, which can be found on the `requirement.txt` file.
2. Run the following command: `python train_minigrid.py --cl --tag=p2e_rs --steps=750000 --seed=6 --plan2explore --expl_intr_scale=0.9 --expl_extr_scale=0.9 --logdir=logs --minlen=5 --rssm_full_recon --sep_exp_eval_policies --reservoir_sampling --recent_past_sampl_thres=0.5`
