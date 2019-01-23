import torch
from torch.distributions import *

m = Normal(0., 1.)
sample1 = m.sample()
print(m.sample())

loss = -m.log_prob(sample1)
print(loss)

print(kl_divergence(m, m))
n = Normal(1., 1.)
print(kl_divergence(m, n))

loc = torch.zeros(3)
scale = torch.ones(3)

# Independent is useful for changing the sahpe of log_prob()
mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
print([mvn.batch_shape, mvn.event_shape])

normal = Normal(loc, scale)
print([normal.batch_shape, normal.event_shape])

diagn = Independent(normal, 1)
print([diagn.batch_shape, diagn.event_shape])