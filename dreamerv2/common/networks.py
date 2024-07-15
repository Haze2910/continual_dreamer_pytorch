import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from dreamerv2 import common

class EnsembleRSSM(nn.Module):
    def __init__(
            self, act_size, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False, embed_size=None,
            act='elu', norm='none', std_act='softplus', min_std=0.1, device="cuda"):
        super().__init__()
        self._device = device
        self._ensemble = ensemble # number of RSSMs
        self._stoch = stoch # size of stochastic state
        self._deter = deter # size of deterministic state (GRU hidden units)
        self._hidden = hidden # size of hidden layer
        self._discrete = discrete # Categorical latent variables, set to False to use Gaussian latent variables (worse)
        self._embed = embed_size
        self._act_size = act_size
        self._act = get_act(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._cell = GRUCell(self._hidden, self._deter, norm=True)
        
        self._obs_out_layers = nn.Sequential(
            nn.Linear(self._deter + self._embed, self._hidden),
            nn.LayerNorm(self._norm) if self._norm != "none" else nn.Identity(),
            self._act
        )
        
        self._img_inp_layers = nn.Sequential(
            nn.Linear(self._discrete * self._stoch + self._act_size if self._discrete else self._stoch + self._act_size, self._hidden),
            nn.LayerNorm(self._norm) if self._norm != "none" else nn.Identity(),
            self._act
        )
        
        self._img_out_layers = nn.ModuleDict()
        for i in range(self._ensemble):
            self._img_out_layers[f"img_out_{i}"] = nn.Sequential(
                nn.Linear(self._deter, self._hidden),
                nn.LayerNorm(self._norm) if self._norm != "none" else nn.Identity(),
                self._act
            )
        
        self._dist_layers = nn.ModuleDict()
        if self._discrete:
            self._dist_layers["obs_dist"] = nn.Linear(self._hidden, self._stoch * self._discrete)  
        else :
            self._dist_layers["obs_dist"] = nn.Linear(self._hidden, 2 * self._stoch)
        for k in range(self._ensemble):
            if self._discrete:
                self._dist_layers[f"img_dist_{k}"] = nn.Linear(self._hidden, self._stoch * self._discrete)
            else:
                self._dist_layers[f"img_dist_{k}"] = nn.Linear(self._hidden, 2 * self._stoch)

    def init_state(self, batch_size):
        deter = torch.zeros(batch_size, self._deter)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]),
                deter=deter)
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]),
                std=torch.zeros([batch_size, self._stoch]),
                stoch=torch.zeros([batch_size, self._stoch]),
                deter=deter)
        state = {key: value.to(self._device) for key, value in state.items()}
        return state

    def get_feat(self, state):
        """
        Combine stochastic and deterministic states
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state['deter']], -1)

    def get_dist(self, state, ensemble=False):
        """
        Get the stochastic state distribution
        """
        if ensemble:
            state = self._suff_stats_ensemble(state['deter'])
        if self._discrete:
            logit = state['logit']
            dist = td.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            dist = td.Normal(mean, std)#dist = tfd.MultivariateNormalDiag(mean, std)  # TODO
        return dist

    def observe(self, embed, action, is_first, state=None):
        """
        Process embeddings and actions to produce prior and posterior states
        """
        swap = lambda x: x.transpose(0, 1)
        if state is None:
            state = self.init_state(action.shape[0])
        post, prior = common.static_scan(
            lambda prev_state, prev_action, embed, is_first: self.obs_step(prev_state[0], prev_action, embed, is_first),
            (swap(action), swap(embed), swap(is_first)), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        
        return post, prior

    def imagine(self, action, state=None):
        """
        Imagine future states given an action and the current state
        """
        swap = lambda x: x.transpose(0, 1)
        if state is None:
            state = self.init_state(action.shape[0])
        assert isinstance(state, dict), state
        action = swap(action)
        prior = common.static_scan(self.img_step, [action], state)[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        def apply_einsum(x, mask):
            return torch.einsum("b,b...->b...", mask, x)
        mask = torch.logical_not(is_first).to(device=self._device)
        prev_state = {key: apply_einsum(value, torch.tensor(mask)) for key, value in prev_state.items()}
        prev_action = apply_einsum(prev_action, torch.tensor(mask))
        
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior['deter'], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state['stoch']
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            prev_stoch = prev_stoch.reshape(shape)
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_inp_layers(x)
        deter = prev_state['deter']
        x, deter = self._cell(x, deter)
        stats = self._suff_stats_ensemble(x)
        index = torch.randint(0, self._ensemble, ())
        stats = {k: v[index] for k, v in stats.items()}
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mean
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_ensemble(self, inputs):
        """
        Compute sufficient statistics (i.e. mean and std  for continuous dists and logits for discrete dists)
        for the distribution representing the stochastic state
        """
        batch_size = list(inputs.shape[:-1])
        inputs = inputs.reshape([-1, inputs.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self._img_out_layers[f"img_out_{k}"](inputs)
            stats.append(self._suff_stats_layer(f"img_dist_{k}", x))
        stats = {k: torch.stack([x[k] for x in stats], 0) for k, v in stats[0].items()}
        stats = {k: v.reshape([v.shape[0]] + batch_size + list(v.shape[2:]))
            for k, v in stats.items()}
        return stats

    def _suff_stats_layer(self, name, x): 
        x = self._dist_layers[name](x)
        if self._discrete:
            logit = x.reshape(x.shape[:-1] + (self._stoch, self._discrete))
            return {'logit': logit}
        else:
            mean, std = torch.chunk(x, 2, -1)
            std = {
                'softplus': lambda: F.softplus(std),
                'sigmoid': lambda: torch.sigmoid(std),
                'sigmoid2': lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        """
        Compute KL Loss
        """
        kl_div = td.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        free = torch.tensor(free)
        if balance == 0.5:
            value = kl_div(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.max(value, free).mean()
        else:
            value_lhs = value = kl_div(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kl_div(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.max(value_lhs.mean(), free)
                loss_rhs = torch.max(value_rhs.mean(), free)
            else:
                loss_lhs = torch.max(value_lhs, free).mean()
                loss_rhs = torch.max(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Encoder(nn.Module):
    """
    Encoder for the observation image
    """
    def __init__(
            self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = [k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        
        if self.cnn_keys:
            _cnn_layers = []
            for i, kernel in enumerate(cnn_kernels):
                input_dim = 2 ** (i - 1) * self._cnn_depth if i > 0 else 3
                depth = 2 ** i * self._cnn_depth
                _cnn_layers.append(nn.Conv2d(input_dim, depth, kernel_size=kernel, stride=2))
                if norm != "none":
                    _cnn_layers.append(nn.LayerNorm(self._norm))
                _cnn_layers.append(self._act)
            self._cnn_layers = nn.Sequential(*_cnn_layers)
        
        if self.mlp_keys:
            _mlp_layers = []
            for i, width in enumerate(mlp_layers):
                _mlp_layers.append(nn.Linear(input_dim, width))
                if norm != "none":
                    _mlp_layers.append(nn.LayerNorm(self._norm))
                _mlp_layers.append(self._act)
                input_dim = width if i == 0 else input_dim
            self._mlp_layers = nn.Sequential(*_mlp_layers)
            
    def __call__(self, data):
        data = {k: data[k] for k in self.cnn_keys}
        x = {k: v.view(-1, *v.shape[-3:]).permute(0, 3, 1, 2) for k, v in data.items()}
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: x[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: x[k] for k in self.mlp_keys}))
        output = torch.cat(outputs, -1)
        return output.reshape(list(data[self.cnn_keys[0]].shape[:-3]) + [output.shape[-1]])

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        for layer in self._cnn_layers:
            x = layer(x)
        return x.reshape(tuple(x.shape[:-3]) + (-1,))

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        for layer in self._mlp_layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """
    Reconstruct the observation images from the latent space as a probability distribution
    """
    def __init__(
            self, input_dim, shapes, cnn_keys=r'.*', mlp_keys=r'.*', act='elu', norm='none',
            cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=[400, 400, 400, 400]):
        super().__init__()
        self._shapes = shapes
        self.cnn_keys = [k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3]
        self.mlp_keys = [k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1]
        self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        
        if self.cnn_keys:
            self._channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
            self._linear_layer = nn.Linear(input_dim, 32 * self._cnn_depth)
            input_dim = 32 * self._cnn_depth
            _cnn_layers = []
            for i, kernel in enumerate(cnn_kernels):
                if i != 0:
                    input_dim = 2 ** (len(cnn_kernels) - (i - 1) - 2) * self._cnn_depth
                depth = 2 ** (len(cnn_kernels) - i - 2) * self._cnn_depth
                if i == len(cnn_kernels) - 1:
                    depth, act, norm = sum(self._channels.values()), nn.Identity(), 'none'
                _cnn_layers.append(nn.ConvTranspose2d(input_dim, depth, kernel_size=kernel, stride=2))
                if norm != "none":
                    _cnn_layers.append(nn.LayerNorm(self._norm))
                _cnn_layers.append(self._act)
            self._cnn_layers = nn.Sequential(*_cnn_layers)
        
        if self.mlp_keys:
            _mlp_layers = []
            for i, width in enumerate(mlp_layers):
                _mlp_layers.append(nn.Linear(input_dim, width))
                if norm != "none":
                    _mlp_layers.append(nn.LayerNorm(self._norm))
                _mlp_layers.append(self._act)
                if i == 0:
                    input_dim = width
            self._mlp_layers = nn.Sequential(*_mlp_layers)
            
            shapes = {k: self._shapes[k] for k in self.mlp_keys}
            _dist_layers = []
            for i, shape in enumerate(shapes.values()):
                _dist_layers.append(DistLayer(input_dim, shape))
                if i == 0:
                    input_dim = shape
            self._dist_layers = nn.Sequential(*_dist_layers)
            
    def __call__(self, features):
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    def _cnn(self, features):
        x = self._linear_layer(features)
        x = x.view(-1, 1, 1, 32 * self._cnn_depth).permute(0, 3, 1, 2)
        for layer in self._cnn_layers:
            x = layer(x)
        x = x.reshape(features.shape[:-1] + x.shape[1:]).permute(0, 1, 3, 4, 2)
        means = torch.split(x, list(self._channels.values()), -1)
        dists = {key: td.Independent(td.Normal(mean, 1), 3)
            for (key, shape), mean in zip(self._channels.items(), means)}
        return dists

    def _mlp(self, features):
        x = features
        for layer in self._mlp_layers:
            x = layer(x)
        dists = {}
        for layer in self._dist_layers:
            x = layer(x)
        return dists

class MLP(nn.Module):
    def __init__(self, input_dim, shape, layers, units, act='elu', norm='none', **out):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._units = units
        self._norm = norm
        self._act = get_act(act)
        self._out = out
        
        _layers = []
        for i in range(layers):
            _layers.append(nn.Linear(input_dim, self._units))
            if self._norm != "none":
                _layers.append(nn.LayerNorm(self._norm))
            _layers.append(self._act)
            if i == 0:
                input_dim = self._units
        _layers.append(DistLayer(input_dim, self._shape, **self._out))
        self._layers = nn.Sequential(*_layers)

    def __call__(self, features):
        x = features
        for layer in self._layers:
            x = layer(x)
        return x


class GRUCell(nn.Module):
    """
    RNN used to capture temporal dependencies of the deterministic state across the time steps.
    GRUcell is used instead of an LSTM because its efficiency.
    """
    def __init__(self, input_dim, size, norm=False, act='tanh', update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = get_act(act)
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(input_dim + size, 3 * size, bias=norm is not None)
        if norm:
            self._norm = nn.LayerNorm(3 * size)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, output


class DistLayer(nn.Module):
    def __init__(self, input_dim, shape, dist='mse', min_std=0.1, init_std=0.0):
        super().__init__()
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._output_linear_layer = nn.Linear(input_dim, int(np.prod(self._shape)))
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            self._dist_linear_layer = nn.Linear(input_dim, int(np.prod(self._shape)))

    def __call__(self, inputs):
        out = self._output_linear_layer(inputs)
        out = out.view(*inputs.shape[:-1], *self._shape)
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal'):
            std = self._dist_linear_layer(inputs)
            std = std.view(*inputs.shape[:-1], *self._shape)
        if self._dist == 'mse':
            dist = td.Normal(out, 1.0)
            return td.Independent(dist, len(self._shape))
        if self._dist == 'normal':
            dist = td.Normal(out, std)
            return td.Independent(dist, len(self._shape))
        if self._dist == 'binary':
            dist = common.Bernoulli(logits=out, validate_args=False)
            return td.Independent(dist, len(self._shape))
        if self._dist == 'tanh_normal':
            mean = 5 * torch.tanh(out / 5)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, common.TanhBijector())
            dist = td.Independent(dist, len(self._shape))
            return common.SampleDist(dist)
        if self._dist == 'trunc_normal':
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(torch.tanh(out), std, -1, 1)
            return td.Independent(dist, 1)
        if self._dist == 'onehot':
            return common.OneHotDist(out)
        raise NotImplementedError(self._dist)

def get_act(name):
    if name == 'none':
        return nn.Identity()
    elif name == 'mish':
        return nn.Mish()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(name)
