import re
import functools

import torch
import torch.optim as optim
import torch.distributions as td

from . import dists


class RandomAgent:
    def __init__(self, act_space, logprob=False):
        self.act_space = act_space['action']
        self.logprob = logprob
        if hasattr(self.act_space, 'n'):
            self._dist = dists.OneHotDist(torch.zeros_like(torch.Tensor(self.act_space.low)))#torch.zeros(self.act_space.n))
        else:
            dist = td.Uniform(torch.Tensor(self.act_space.low), torch.Tensor(self.act_space.high))#self.act_space.low, self.act_space.high)
            self._dist = td.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample((len(obs['is_first']),))
        output = {'action': action}
        if self.logprob:
            output['logprob'] = self._dist.log_prob(action)
        return output, None

class CarryOverState:

    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out

@functools.total_ordering
class Counter:
    def __init__(self, initial=0):
        self.value = initial
    def __int__(self):
        return int(self.value)

    def __eq__(self, other):
        return int(self) == other

    def __ne__(self, other):
        return int(self) != other

    def __lt__(self, other):
        return int(self) < other

    def __add__(self, other):
        return int(self) + other

    def increment(self, amount=1):
        self.value += amount

class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        step = int(step)
        if not self._every:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False

class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        step = int(step)
        if not self._until:
            return True
        return step < self._until

def static_scan(fn, inputs, start, reverse=False):
    """
    
    """
    last = start
    indices = reversed(range(inputs[0].shape[0])) if reverse else range(inputs[0].shape[0])
    outputs = None
    for index in indices:
        input = (inp[index] for inp in inputs)
        last = fn(last, *input)
        if outputs is None:
            if isinstance(last, dict):
                outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
            else:
                outputs = [item.clone().unsqueeze(0) if not isinstance(item, dict) else {key: value.clone().unsqueeze(0) for key, value in item.items()} for item in last]
        else:
            if isinstance(last, dict):
                for key, value in last.items():
                    outputs[key] = torch.cat([outputs[key], value.unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for key, value in last[j].items():
                            outputs[j][key] = torch.cat([outputs[j][key], value.unsqueeze(0)], dim=0)
                    else:
                        outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
    if reverse:
        if isinstance(outputs, list):
            outputs = [list(reversed(x)) for x in outputs]
        else:
            outputs = {key: list(reversed(value)) for key, value in outputs.items()}
    return [outputs] if isinstance(last, dict) else outputs

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap)
    if axis != 0:
        returns = returns.permute(dims)
    return returns

def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        step = torch.tensor(step, dtype=torch.float32)
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)

def action_noise(action, amount, act_space):
    if amount == 0:
        return action
    if hasattr(act_space, 'n'):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return torch.clamp(td.Normal(action, amount).sample(), -1, 1)

class StreamNorm(torch.nn.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        super().__init__()
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = torch.nn.Parameter(torch.ones(shape, dtype=torch.float64), requires_grad=False)

    def __call__(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics['mean'] = inputs.mean()
        metrics['std'] = inputs.std()
        outputs = self.transform(inputs)
        metrics['normed_mean'] = outputs.mean()
        metrics['normed_std'] = outputs.std()
        return outputs, metrics

    def reset(self):
        self.mag.data.fill(1.0)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = batch.abs().mean(0, dtype=torch.float64)
        self.mag.data.mul_(self._momentum).add_((1 - self._momentum) * mag)

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= (self.mag.to(inputs.dtype) + self._eps).unsqueeze(0)
        values *= self._scale
        return values.reshape(inputs.shape)

class Optimizer():
    def __init__(self, name, parameters, lr, eps=1e-4, clip=None, wd=None,
                opt='adam', wd_pattern=r'.*', enable_fp16=False):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            'adam': lambda: optim.Adam(parameters, lr=lr, eps=eps),
            'adamax': lambda: optim.Adamax(parameters, lr=lr, eps=eps),
            'sgd': lambda: optim.SGD(parameters, lr=lr),
            'momentum': lambda: optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=enable_fp16)

    def __call__(self, loss, parameters, retain_graph=False):
        assert len(loss.shape) == 0, (self._name, loss.shape)
        metrics = {}
        metrics[f'{self._name}_loss'] = loss.detach().item()

        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)

        # Gradient clipping.
        if self._clip:
            self._scaler.unscale_(self._opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self._clip)
            metrics[f'{self._name}_grad_norm'] = grad_norm.item()
            
        # Weight Decay
        if self._wd:
            self._apply_weight_decay(parameters)

        self._scaler.step(self._opt)
        self._scaler.update()
        
        return metrics

    def _apply_weight_decay(self, parameters):
        nontrivial = (self._wd_pattern != r'.*')
        if nontrivial:
            print('Applied weight decay to variables:')
        for param in parameters:
            if re.search(self._wd_pattern, self._name + '/' + param.name):
                if nontrivial:
                    print('- ' + self._name + '/' + param.name)
                param.assign((1 - self._wd) * param)

class RequiresGrad:
  def __init__(self, model):
    self._model = model

  def __enter__(self):
    self._model.requires_grad_(requires_grad=True)

  def __exit__(self, *args):
    self._model.requires_grad_(requires_grad=False)