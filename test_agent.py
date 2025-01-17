import torch
import numpy as np
import gymnasium as gym
import minigrid
import warnings
import argparse
from dreamerv2 import common
from dreamerv2.agent import Agent
from ruamel.yaml import YAML
import pathlib
from gymnasium.wrappers import RecordVideo

warnings.filterwarnings('ignore', category=UserWarning)

# Parser for recording a video
parser = argparse.ArgumentParser()
parser.add_argument("--record", action="store_true")
args = parser.parse_args()

# Load configs
yaml = YAML(typ="safe", pure=True)
config = yaml.load((pathlib.Path('test\config.yaml')).read_text())
config = common.Config(config)

env_names = [
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-LavaCrossingS9N1-v0',
    'MiniGrid-SimpleCrossingS9N1-v0',
]

num_tasks = len(env_names)
envs = []
for i in range(num_tasks):
    name = env_names[i]
    env = gym.make(name, render_mode="rgb_array" if args.record else "human")
    if args.record:
        env = RecordVideo(env, video_folder=f"test/videos", name_prefix=i)
    env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
    env = common.GymWrapper(env) 
    env = common.ResizeImage(env) 
    env = common.OneHotAction(env) if hasattr(env.act_space['action'], 'n') else common.NormalizeAction(env) 
    env = common.TimeLimit(env, config.time_limit)
    envs.append(env)


task_id = 0
while task_id < len(envs):
    print(f"\nTask {task_id}")
    env = envs[task_id]
    agent = Agent(config, env.obs_space, env.act_space)
    agent.requires_grad_(requires_grad=False)
    agent.load_state_dict(torch.load('test\model.pt'))
    
    driver = common.Driver([env], cl=config.cl)
    
    def on_episode(ep, task_idx):
        score = float(ep['reward'].astype(np.float64).sum())
        print(f"Score {score}")
                
    driver.on_episode(on_episode)
    policy = lambda *args: agent.policy(*args, mode='eval')
    driver(policy, episodes=1)
    
    task_id += 1



