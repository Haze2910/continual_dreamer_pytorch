import torch

from dreamerv2 import agent
from dreamerv2 import common


class Plan2Explore(torch.nn.Module):
    """
    Exploration strategy that combines intrinsic and extrinsic (environment) rewards.
    The intrisic rewards motivate the agent to visit new or uncertain states.
    An ActorCritic model is used to learn the policy
    """
    def __init__(self, config, act_space, wm, reward):
        super().__init__()
        self.config = config
        self.enable_fp16 = True if config.precision == 16 else False
        self.reward = reward
        self.wm = wm
        self.ac = agent.ActorCritic(config, act_space)
        self.actor = self.ac.actor
        stoch_size = config.rssm.stoch
        
        if config.rssm.discrete:
            stoch_size *= config.rssm.discrete
        size = {
            'embed': 32 * config.encoder.cnn_depth,
            'stoch': stoch_size, # default disagreement target
            'deter': config.rssm.deter,
            'feat': config.rssm.stoch + config.rssm.deter,
        }[self.config.disag_target]
        if config.rssm.discrete:
            input_size = config.rssm.stoch * config.rssm.discrete + config.rssm.deter
        else:
            input_size = config.rssm.stoch + config.rssm.deter
        if self.config.disag_action_cond:
            act_size = act_space.n if hasattr(act_space, "n") else act_space[0]
            input_size += act_size
            
        # Ensemble of MLP networks for intrinsic reward prediction
        self._networks = torch.nn.ModuleList([common.MLP(input_dim=input_size, shape=size, **config.expl_head)
                                              for _ in range(config.disag_models)])
        self.opt = common.Optimizer('expl', self.parameters(), **config.expl_opt)
        self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
        self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

    def train(self, start, context, data):
        metrics = {}
        stoch = start['stoch']
        if self.config.rssm.discrete:
            stoch = stoch.view(list(stoch.shape[:-2]) + [stoch.shape[-2] * stoch.shape[-1]])
        target = {
            'embed': context['embed'],
            'stoch': stoch, # default
            'deter': start['deter'],
            'feat': context['feat'],
        }[self.config.disag_target]
        inputs = context['feat']
        if self.config.disag_action_cond:
            action = torch.tensor(data['action']).to(self.config.device)
            inputs = torch.cat([inputs, action], -1)
        # First train the ensemble, then the ActorCritic
        metrics.update(self._train_ensemble(inputs, target))
        metrics.update(self.ac.train(self.wm, start, data['is_terminal'], self._compute_reward))
        return None, metrics

    def _compute_reward(self, seq):
        """
        Compute the intrinsic reward as the std of the disagreement between
        each head of the ensemble and evenutally combine it to the extrinsic reward
        """
        inputs = seq['feat']
        if self.config.disag_action_cond:
            action = seq['action'].to(inputs.dtype)
            inputs = torch.cat([inputs, action], -1)
        
        # Stack the reward prediction for each network in the ensemble
        preds = [head(inputs).mode for head in self._networks] 
        preds = torch.stack(preds)
        
        # Compute the disagreement between the predictions as their standard deviation, 
        # the greater the disagreement, the uncertain is the prediction
        disagreement = preds.std(0).mean(-1)
        
        # Compute the final intrinsic reward (lognormalize and scale)
        if self.config.disag_log:
            disagreement = torch.log(disagreement)
        reward = self.config.expl_intr_scale * self.intr_rewnorm(disagreement)[0]
        
        # Combine the intrinsic and extrinsic reward 
        if self.config.expl_extr_scale:
            reward += self.config.expl_extr_scale * self.extr_rewnorm(self.reward(seq))[0]
        return reward

    def _train_ensemble(self, inputs, targets):
        """
        The ensemble is trained to produce similar outputs, using a negative log-likelihood loss function
        """
        if self.config.disag_offset:
            targets = targets[:, self.config.disag_offset:]
            inputs = inputs[:, :-self.config.disag_offset]
        targets = targets.detach()
        inputs = inputs.detach()
        with common.RequiresGrad(self):
            with torch.cuda.amp.autocast(self.enable_fp16):
                preds = [head(inputs) for head in self._networks]
                loss = -sum([pred.log_prob(targets).mean() for pred in preds])
            metrics = self.opt(loss, self.parameters())
        return metrics

