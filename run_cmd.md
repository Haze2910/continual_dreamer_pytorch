python main.py --logdir ./logdir/atari_pong --configs defaults atari --task atari_ALE/Pong-v5

python run_minigrid.py --logdir ./logdir/minigrid --configs defaults minigrid

# ORIGINAL
python train_minigrid.py --cl --tag=p2e_rs --steps=750000 --seed=6 --plan2explore --expl_intr_scale=0.9 --expl_extr_scale=0.9 --logdir=logs --minlen=5 --rssm_full_recon --sep_exp_eval_policies --reservoir_sampling --recent_past_sampl_thres=0.5
