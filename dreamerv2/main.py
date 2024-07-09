import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import numpy as np
from ruamel.yaml import YAML

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import torch
from agent import Agent
from expl import Plan2Explore
import common

# Load configs
yaml = YAML(typ="safe", pure=True)
configs = yaml.load((pathlib.Path(__file__).parent/'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))

def cl_train_loop(envs, config, eval_envs=None):

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)

    # Configs
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir/'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    replay = common.Replay(logdir/'train_episodes', **config.replay, num_tasks=config.num_tasks)

    total_step = common.Counter(replay.stats['total_steps'])
    print("Replay buffer total steps: {}".format(replay.stats['total_steps']))
    
    logger = common.Logger(total_step, logdir=config.logdir, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)
    replay.logger = logger
    
    # Retrieve the current step and task from stopped run
    task_id = int(replay.stats['total_steps'] // config.steps)
    print("Task {}".format(task_id))
    restart_step = int(replay.stats['total_steps'] % config.steps)
    print("Restart step: {}".format(restart_step))
    restart = True if restart_step > 0 else False

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    if config.expl_every:
        print("exploring every {} steps".format(config.expl_every))
        should_expl = common.Every(config.expl_every)
    else:
        should_expl = common.Until(config.expl_until)

    def on_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        logger.scalar('return', score)
        logger.scalar('length', length)
        logger.scalar('task', task_id)
        logger.scalar('replay_capacity', replay.stats['loaded_steps'])
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
        logger.add(replay.stats)
        logger.write()

    def create_envs_drivers(env):
        env = common.GymWrapper(env)
        env = common.ResizeImage(env)
        env = common.OneHotAction(env) if hasattr(env.act_space['action'], 'n') else common.NormalizeAction(env)
        env = common.TimeLimit(env, config.time_limit)

        driver = common.Driver([env])
        driver.on_episode(on_episode)
        driver.on_step(replay.add_step)
        driver.on_reset(replay.add_step)
        driver.on_step(lambda tran, worker: total_step.increment())
        driver.on_step(lambda tran, worker: step.increment())
        return env, driver

    if eval_envs is None:
        eval_envs = envs

    _eval_envs = []
    for i in range(len(eval_envs)):
        env, _ = create_envs_drivers(eval_envs[i])
        _eval_envs.append(env)

    try:
        while task_id < len(envs):
            print(f"\n\t Task {task_id} \n")

            env = envs[task_id]
            if restart:
                start_step = restart_step
                restart = False
            else:
                start_step = 0
            
            replay.set_task(task_id) # set current task id in the buffer to save with the episode
            step = common.Counter(start_step) # steps of current task

            env, driver = create_envs_drivers(env)

            # Prefill the buffer if empty to avoid errors when sampling
            prefill = max(0, config.prefill - replay.stats['total_steps'])
            if prefill:
                print(f'Prefill dataset ({prefill} steps).')
                random_agent = common.RandomAgent(env.act_space)
                driver(random_agent, steps=prefill, episodes=1)
                driver.reset()

            print('Create agent.')
            agent = Agent(config, env.obs_space, env.act_space, total_step)
            agent.requires_grad_(requires_grad=False) # by default grad is disabled, we will enable it during training when necessary

            if isinstance(agent._expl_behavior, Plan2Explore):
                replay.agent = agent

            # Dataset: {batch_size: 16, seq_length: 50}
            dataset = iter(replay.dataset(**config.dataset))
            train_agent = common.CarryOverState(agent.train) # save current state for next train call
            train_agent(next(dataset))
            
            # Load saved model if exists
            if (logdir/'latest_model.pt').exists():
                print("Loading agent.")
                agent.load_state_dict(torch.load(logdir/'latest_model.pt'))
            else:
                print('Pretrain agent.')
                for _ in range(config.pretrain):
                    train_agent(next(dataset))
            
            policy = lambda *args: agent.policy(*args, mode='explore' if should_expl(total_step) else 'train')

            def eval_on_episode(ep, task_idx):
                length = len(ep['reward']) - 1
                score = float(ep['reward'].astype(np.float64).sum())
                logger.scalar('eval_return_{}'.format(task_idx), score)
                logger.scalar('eval_length_{}'.format(task_idx), length)
                ep = {k: np.expand_dims(v, axis=0) for k, v in ep.items()}
                logger.write()

            def train_step(transition, worker):
                if should_train(total_step):
                    for _ in range(config.train_steps):
                        mets = train_agent(next(dataset))
                        [metrics[key].append(value) for key, value in mets.items()]
                if should_log(total_step):
                    for name, values in metrics.items():
                        logger.scalar(name, float(np.mean(values)))
                        metrics[name].clear()
                    logger.write()

            driver.on_step(train_step)

            eval_driver = common.Driver(_eval_envs, cl=config.cl)
            eval_driver.on_episode(eval_on_episode)
            eval_policy = lambda *args: agent.policy(*args, mode='eval')

            # Training loop
            while step < config.steps:
                logger.write()
                driver(policy, steps=config.eval_every)
                if config.sep_exp_eval_policies:
                    eval_driver(eval_policy, steps=config.eval_steps)
                else:
                    eval_driver(policy, steps=config.eval_steps)
                torch.save(agent.state_dict(), logdir/'latest_model.pt')
            
            # Increment the task id
            task_id += 1

    except KeyboardInterrupt: 
        print("Keyboard Interrupt, saving the agent...")
        torch.save(agent.state_dict(), logdir/"latest_model.pt")
