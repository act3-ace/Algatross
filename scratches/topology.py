import torch
from algatross.algorithms.genetic.mo_aim.topology import MOAIMTopology


t = MOAIMTopology(alpha=1)
t.set_archipelago(list(range(7)), list(range(7, 12)))
print(t)
t.optimize_softmax(torch.rand((5, 7)))
print(t)
