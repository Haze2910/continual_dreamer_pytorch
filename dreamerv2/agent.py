import torch
import torch.nn as nn

#from p2e import Plan2Explore
from dreamerv2 import p2e
from dreamerv2 import common

to_np = lambda x: x.detach().cpu().numpy()

class Agent(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.config = config
        self.enable_fp16 = True if config.precision == 16 else False # enable mixed precision for a faster training
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.wm = WorldModel(config, self.act_space, obs_space).to(self.config.device)
        self._task_behavior = ActorCritic(config, self.act_space).to(self.config.device)
        if config.expl_behavior == 'greedy':
            self._expl_behavior = self._task_behavior.to(self.config.device)
        else:
            self._expl_behavior = p2e.Plan2Explore(self.config, self.act_space, self.wm,
                                               lambda sequence: self.wm.heads['reward'](sequence['feat']).mode).to(self.config.device)

    def policy(self, obs, state=None, mode='train'):
        """
        Computes the policy action based on the current observation and state.

        Inputs:
            obs (dict): Current observation.
            state (tuple): Current state (latent state, action).
            mode (str): One in ['train', 'eval', 'explore'].

        Output:
            tuple: (outputs, state) where outputs is a dict with the action and state is the next state.
        """
        obs = {k: torch.tensor(v) for k, v in obs.items()}
        
        with torch.cuda.amp.autocast(enabled=self.enable_fp16):
            if state is None:
                latent = self.wm.rssm.init_state(len(obs['reward']))
                action = torch.zeros((len(obs['reward']),) + self.act_space.shape).to(self.config.device)
                state = latent, action
                
            latent, action = state
            embed = self.wm.encoder(self.wm.preprocess(obs))
            sample = (mode == 'train') or not self.config.eval_state_mean
            latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], sample)
            feat = self.wm.rssm.get_feat(latent)
            
            if mode == 'eval':
                actor = self._task_behavior.actor(feat)
                action = actor.mode()
                noise = self.config.eval_noise
            elif mode in ['explore', 'train']:
                actor = self._expl_behavior.actor(feat) if mode == 'explore' else self._task_behavior.actor(feat)
                action = actor.sample()
                noise = self.config.expl_noise
                
            action = common.action_noise(action, noise, self.act_space)
            outputs = {'action': action}
            state = (latent, action)
            
        return outputs, state
    
    def train(self, data, state=None):
        """
        Train the world model, ac model and exploration model if enabled
        
        Inputs:
            data (dict): dict with batched observation
            state (dict): dict with stochastic state, deterministic state and logit
            
        Outputs:
            state (dict): output state dict with stochastic state, deterministic state and logit 
            metrics (dict): training metrics
        """
        metrics = {}
        
        # First train the world model
        state, outputs, wm_metrics = self.wm.train(data, state)
        metrics.update(wm_metrics)
        
        # Then, the actor-critic model
        start = outputs['post']
        reward = lambda sequence: self.wm.heads['reward'](sequence['feat']).mode
        metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], reward))
        
        # Finally, the exploration model if enabled
        if self.config.expl_behavior != 'greedy':
            metrics.update(self._expl_behavior.train(start, outputs, data)[-1])
            
        return state, metrics


class WorldModel(nn.Module):
    def __init__(self, config, act_space, obs_space):
        super().__init__()
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.enable_fp16 = True if config.precision == 16 else False
        
        feat_size = config.rssm.deter
        feat_size += config.rssm.stoch * config.rssm.discrete if config.rssm.discrete else config.rssm.stoch
        act_size = act_space.n if hasattr(act_space, "n") else act_space[0]
        assert obs_space["image"].shape[:2] == (64, 64), "Image size must be (64, 64)"
        embed_size = 4 * (2 ** (len(config.encoder.cnn_kernels) - 1) * config.encoder.cnn_depth)
        
        self.rssm = common.EnsembleRSSM(act_size, embed_size=embed_size, **config.rssm, device=config.device)
        self.encoder = common.Encoder(shapes, **config.encoder)
        self.heads = nn.ModuleDict({
            "decoder" : common.Decoder(feat_size, shapes, **config.decoder),
            "reward" : common.MLP(feat_size, [], **config.reward_head),
            "discount" : common.MLP(feat_size, [], **config.discount_head)
        })
                   
        self.model_opt = common.Optimizer('model', self.parameters(), **config.model_opt, enable_fp16=self.enable_fp16)

    def train(self, data, state=None):
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(self.enable_fp16):
                model_loss, state, outputs, metrics = self.loss(data, state)
            metrics.update(self.model_opt(model_loss, self.parameters()))
        return state, outputs, metrics
    
    def loss(self, data, state=None):
        # Preprocess and encode
        data = self.preprocess(data)
        embed = self.encoder(data)
        
        # Observe rollout and compute kl divergence loss between post and prior
        post, prior = self.rssm.observe(embed, data['action'], data['is_first'], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        losses = {'kl': kl_loss}
        
        # Compute image, reward and discount loss
        likes = {}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            input = feat if grad_head else feat.detach()
            out = head(input)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()
                
        # Aggregate all the losses
        model_loss = sum(self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        metrics = {f'{name}_loss': to_np(value) for name, value in losses.items()}
        
        post = {key: value.detach() for key, value in post.items()}
        outs = dict(embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value)
        last_state = {k: v[:, -1] for k, v in post.items()}
        
        # Save some metrics
        metrics['model_kl'] = to_np(kl_value.mean())
        metrics['prior_ent'] = to_np(self.rssm.get_dist(prior).entropy().mean())
        metrics['post_ent'] = to_np(self.rssm.get_dist(post).entropy().mean())
        return model_loss, last_state, outs, metrics

    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:])) # flatten first two dims (bs, seq_len)
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.rssm.get_feat(start) # latent state
        start['action'] = torch.zeros_like(policy(start['feat']).mode())
        sequence = {k: [v] for k, v in start.items()}
        
        for _ in range(horizon):
            action = policy(sequence['feat'][-1].detach()).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in sequence.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                sequence[key].append(value)
        sequence = {k: torch.stack(v, 0) for k, v in sequence.items()}
        
        if 'discount' in self.heads:
            discount = self.heads['discount'](sequence['feat']).mean
            if is_terminal is not None:
                # Override discount prediction for the first step with the true discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal)
                true_first *= self.config.discount
                discount = torch.cat([torch.tensor(true_first).to(self.config.device)[None], discount[1:]], 0)
        else:
            discount = self.config.discount * torch.ones(sequence['feat'].shape[:-1])
        sequence['discount'] = discount
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        sequence['weight'] = torch.cumprod(torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0)
        return sequence

    def preprocess(self, obs):
        obs = obs.copy()
        obs = {k: torch.Tensor(v).to(self.config.device) for k, v in obs.items()}
        obs["image"] = obs["image"] / 255.0 - 0.5 # Normalize
        obs['reward'] = {
            'identity': lambda x: x,
            'sign': torch.sign,
            'tanh': torch.tanh,
        }[self.config.clip_rewards](obs['reward']) # Clip rewards
        obs['discount'] = torch.logical_not(obs['is_terminal']).float() # Set discount to zero if done
        obs['discount'] *= self.config.discount # Update discount
        return obs


class ActorCritic(nn.Module):
    def __init__(self, config, act_space):
        super().__init__()
        self.config = config
        self.enable_fp16 = True if config.precision == 16 else False
        self.act_space = act_space
        
        discrete = hasattr(act_space, 'n')
        if self.config.actor.dist == 'auto':
            self.config = self.config.update({'actor.dist': 'onehot' if discrete else 'trunc_normal'})
        if self.config.actor_grad == 'auto':
            self.config = self.config.update({'actor_grad': 'reinforce' if discrete else 'dynamics'})
            
        if config.rssm.discrete:
            feat_size = config.rssm.stoch * config.rssm.discrete + config.rssm.deter
        else:
            feat_size = config.rssm.stoch + config.rssm.deter
        self.actor = common.MLP(feat_size, act_space.shape[0], **self.config.actor)
        self.critic = common.MLP(feat_size, [], **self.config.critic)
        if self.config.slow_target:
            self._target_critic = common.MLP(feat_size, [], **self.config.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
            
        self.actor_opt = common.Optimizer('actor', self.actor.parameters(), **self.config.actor_opt, enable_fp16=self.enable_fp16)
        self.critic_opt = common.Optimizer('critic', self.critic.parameters(), **self.config.critic_opt, enable_fp16=self.enable_fp16)
        self.reward_norm = common.StreamNorm(**self.config.reward_norm)

    def train(self, world_model, start, is_terminal, reward_fn):
        """
        Imagine trajectories using the world model and the actor
        """
        metrics = {}
        horizon = self.config.imag_horizon

        # Forward passes
        with common.RequiresGrad(self.actor): # actor grad enabled, critic disabled
            with torch.cuda.amp.autocast(self.enable_fp16):
                # Imagine
                sequence = world_model.imagine(self.actor, start, is_terminal, horizon)
                reward = reward_fn(sequence)
                sequence['reward'], reward_metrics = self.reward_norm(reward)
                reward_metrics = {f'reward_{k}': to_np(v) for k, v in reward_metrics.items()}
                
                # Compute target and losses
                target, target_metrics = self.target(sequence)
                actor_loss, actor_metrics = self.actor_loss(sequence, target)        
        with common.RequiresGrad(self.critic): # actor grad disabled, critc enabled
            with torch.cuda.amp.autocast(self.enable_fp16):
                critic_loss, critic_metrics = self.critic_loss(sequence, target)
        
        # Backward passes
        with common.RequiresGrad(self):
            metrics.update(self.actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self.critic_opt(critic_loss, self.critic.parameters()))
        
        # Update metrics
        metrics.update(**reward_metrics, **target_metrics, **actor_metrics, **critic_metrics)
        self.update_slow_target()
        return metrics

    def actor_loss(self, sequence, target):
        """
        Compute actor loss weigthed by the trajectory weights
        """
        metrics = {}
        # Last two states are removed, one for bootstrapping and the other because it's the last action
        # which doesn't have a following state
        policy = self.actor(sequence['feat'][:-2].detach()) 
        if self.config.actor_grad == 'dynamics':
            objective = target[1:]
        elif self.config.actor_grad == 'reinforce':
            baseline = self._target_critic(sequence['feat'][:-2]).mode
            advantage = (target[1:] - baseline).detach()
            # First action is removed because the initial state comes from the replay buffer
            objective = policy.log_prob(sequence['action'][1:-1]) * advantage 
        entropy = policy.entropy()
        entropy_scale = self.config.actor_ent
        objective += entropy_scale * entropy
        weight = sequence['weight'].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics['actor_ent'] = to_np(entropy.mean())
        metrics['actor_ent_scale'] = entropy_scale
        return actor_loss, metrics

    def critic_loss(self, sequence, target):
        """
        Compute the critic loss as the negative log probability of the target values
        """
        dist = self.critic(sequence['feat'][:-1].detach())
        target = target.detach()
        weight = sequence['weight'].detach()
        critic_loss = -torch.mean(dist.log_prob(target) * weight[:-1])
        metrics = {'critic': to_np(dist.mode.mean())}
        return critic_loss, metrics

    def target(self, sequence):
        """
        Compute the target values for the critic to predict using reward and discounted future values
        """
        reward = sequence['reward']
        discount = sequence['discount']
        value = self._target_critic(sequence['feat']).mode
        # Skipping last time step because it is used for bootstrapping.
        target = common.lambda_return(
            reward[:-1], value[:-1], discount[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0)
        target = torch.stack(target, dim=1)
        metrics = {}
        metrics['critic_slow'] = to_np(value.mean())
        metrics['critic_target'] = to_np(value.mean())
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(self.config.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s + (1 - mix) * d
            self._updates += 1
            
