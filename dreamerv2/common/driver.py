import numpy as np

class Driver:
    def __init__(self, envs, cl=False):
        self._envs = envs
        self._cl = cl
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            # Reset envs if necessary
            obs = {i: self._envs[i].reset() for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
            
            # Call reset callbacks
            for i, ob in obs.items():
                self._obs[i] = ob
                action = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()} # dummy action to store the transition
                transition = {k: self._convert(v) for k, v in {**ob, **action}.items()}
                for fn in self._on_resets:
                    fn(transition, worker=i)
                self._eps[i] = [transition]
            
            # Get actions and next states from policy
            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            actions, self._state = policy(obs, self._state)
            actions = [{k: np.array(actions[k][i].cpu().numpy()) for k in actions} for i in range(len(self._envs))]

            # Perform actions in the envs
            obs = []
            for e, a in zip(self._envs, actions):
                try:
                    ob = e.step(a)
                except RuntimeError: # sometimes the minigrid env fails and needs a reset
                    ob = e.reset()
                obs.append(ob)
            obs = [ob for ob in obs]
            
            # Call step callbacks
            for i, (action, ob) in enumerate(zip(actions, obs)):
                transition = {k: self._convert(v) for k, v in {**ob, **action}.items()}
                for fn in self._on_steps:
                    fn(transition, worker=i)
                self._eps[i].append(transition)
                step += 1
                
                # If the observation is terminal call the episode callbacks
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    for fn in self._on_episodes:
                        if self._cl:
                            fn(ep, task_idx=i) 
                        else:
                            fn(ep)   
                    episode += 1
            self._obs = obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
