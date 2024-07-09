import torch
import argparse
import gymnasium as gym
import minigrid
import dreamerv2.main as dv2

# Check GPU
if torch.cuda.is_available():
    print("CUDA available. Number of GPUs:", torch.cuda.device_count())
    device = torch.device('cuda:0')
else:
    print("CUDA not available. Using CPU.")
    device = torch.device('cpu')

def main(args):
    
    # Define the envs to sequentially train the agent on, we'll train on each of them a fixed amount of steps
    env_names = [
        'MiniGrid-DoorKey-8x8-v0',
        'MiniGrid-LavaCrossingS9N1-v0',
        'MiniGrid-SimpleCrossingS9N1-v0',
        'MiniGrid-MultiRoom-N6-v0',
    ]
    
    num_tasks = len(env_names)
    tag = args.tag + "_" + str(args.seed)
    config = dv2.defaults.update({
        'logdir': f'{args.logdir}/minigrid_{tag}',
        'log_every': 1e3,
        'log_every_video': 2e5,
        'train_every': 10,
        'prefill': 1e4,
        'time_limit': 100,  
        'actor_ent': 3e-3,
        'loss_scales.kl': 1.0,
        'discount': 0.99,
        'steps': args.steps,
        'cl': args.cl,
        'num_tasks': num_tasks,
        'seed': args.seed,
        'eval_every': 1e4,
        'eval_steps': 1e3,
        'tag': tag,
        'replay.capacity': args.replay_capacity,
        'replay.reservoir_sampling': args.reservoir_sampling,
        'replay.recent_past_sampl_thres': args.recent_past_sampl_thres,
        'sep_exp_eval_policies': args.sep_exp_eval_policies,
        'replay.minlen': args.minlen,
    }).parse_flags()

    if args.plan2explore:
        config = config.update({
            'expl_behavior': 'Plan2Explore',
            'expl_intr_scale': args.expl_intr_scale,
            'expl_extr_scale': args.expl_extr_scale,
            'expl_every': args.expl_every,
        }).parse_flags()

    envs, eval_envs = [], []
    for i in range(num_tasks):
        name = env_names[i]
        env = gym.make(name)
        env = minigrid.wrappers.RGBImgPartialObsWrapper(env) # the agent sees only a portion of environment (can't see behind)
        envs.append(env)
        eval_env = gym.make(name)
        eval_env = minigrid.wrappers.RGBImgPartialObsWrapper(eval_env)
        eval_envs.append(eval_env)

    dv2.cl_train_loop(envs, config, eval_envs=eval_envs)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Dreamer Minigrid")

    # General
    parser.add_argument('--cl', default=False, action='store_true', help='Flag for continual learning loop.')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5e5)
    parser.add_argument('--env', type=int, default=0, help='picks the env for the single task dv2.')
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--tag', type=str, default='', help='unique str to tag tb.')
    parser.add_argument('--logdir', type=str, default='logs', help='directory for the tb logs and exp replay episodes.')
    
    # Buffer settings
    parser.add_argument('--replay_capacity', type=int, default=2e6)
    parser.add_argument('--reservoir_sampling', default=False, action='store_true', help='Flag for using reservoir sampling.') 
    parser.add_argument('--recent_past_sampl_thres', type=float, default=0., help="probability of triangle distribution, expected to be > 0 and <= 1. 0 denotes taking episodes always from uniform distribution.")
    parser.add_argument('--minlen', type=int, default=50, help="minimal episode length of episodes stored in the replay buffer")
    parser.add_argument('--batch_size', type=int, default=16, help="mini-batch size")

    # Exploration
    parser.add_argument('--plan2explore', default=False, action='store_true', help='Enable plan2explore exploration strategy.')
    parser.add_argument('--expl_intr_scale', type=float, default=1.0, help="scale of the intrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_extr_scale', type=float, default=0.0, help="scale of the extrinsic reward needs to be > 0 and <= 1.")
    parser.add_argument('--expl_every', type=int, default=0,  help="how often to run the exploration strategy.")
    parser.add_argument('--sep_exp_eval_policies', default=False, action='store_true', help='Whether to use separate exploration and evaluation polcies.')
    
    args = parser.parse_known_args(args=None)[0]
    main(args)


