import json
import torch

# Distributions: sample, log_prob, cdf, icdf
from torch.distributions import Normal, Uniform, Bernoulli, Categorical

# Normal distribution
normal = Normal(0.0, 1.0)
samples = [normal.sample() for _ in range(5)]
log_probs = [normal.log_prob(s) for s in samples]

# Uniform distribution
uniform = Uniform(0.0, 10.0)
uniform_samples = [uniform.sample() for _ in range(3)]

# Bernoulli distribution
bernoulli = Bernoulli(0.7)
bern_samples = [bernoulli.sample() for _ in range(5)]

# Categorical distribution
cat = Categorical(torch.tensor([0.1, 0.2, 0.7]))
cat_samples = [cat.sample() for _ in range(5)]

out = {
    "normal_mean": normal.mean.tolist(),
    "normal_samples": [s.tolist() for s in samples],
    "uniform_samples": [s.tolist() for s in uniform_samples],
    "bernoulli_samples": [s.tolist() for s in bern_samples],
    "categorical_probs": cat.probs.tolist(),
    "categorical_samples": [int(s) for s in cat_samples],
    "status": "OK"
}
print(json.dumps(out, indent=2))
