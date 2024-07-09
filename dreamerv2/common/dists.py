import torch
import torch.nn.functional as F
import torch.distributions as td
import numpy as np


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample((self._samples,))
        return samples.mean(0)

    def mode(self):
        sample = self._dist.sample((self._samples,))
        logprob = self._dist.log_prob(sample)
        return sample[logprob.argmax()]

    def entropy(self):
        sample = self._dist.sample((self._samples,))
        logprob = self.log_prob(sample)
        return -logprob.mean(0)


class OneHotDist(td.OneHotCategorical):
    def __init__(self, logits=None, probs=None, dtype=None):
        super().__init__(logits=logits, probs=probs)
        
    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError('need to check')
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample
    
class TruncatedNormal(td.TransformedDistribution):
    def __init__(self, loc, scale, low, high):
        self.base_dist = td.Normal(loc, scale)
        self.low = low
        self.high = high
        super().__init__(self.base_dist, td.transforms.ComposeTransform([
            td.transforms.AffineTransform(loc, scale),
            td.transforms.SigmoidTransform(),
            td.transforms.AffineTransform(low, high - low),
        ]))

    def sample(self, sample_shape=torch.Size()):
        samples = super().sample(sample_shape)
        samples = torch.clamp(samples, self.low, self.high)
        return samples

    def rsample(self, sample_shape=torch.Size()):
        samples = super().rsample(sample_shape)
        samples = torch.clamp(samples, self.low, self.high)
        return samples

class TruncNormalDist(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

    def sample(self, *args, **kwargs):
        event = super().sample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(td.transforms.Transform):
    def __init__(self):
        super(TanhBijector, self).__init__()

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = y.clamp(-0.99999997, 0.99999997)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (np.log(2.0) - x - torch.nn.functional.softplus(-2.0 * x))
    
    
class Bernoulli(td.Bernoulli):
    @property
    def mode(self):
        return torch.round(self.probs)
    