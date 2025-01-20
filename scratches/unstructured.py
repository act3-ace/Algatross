from algatross.quality_diversity.archives.unstructured import UnstructuredArchive
import numpy as np

rng = np.random.default_rng(100)
ball_arch = UnstructuredArchive(solution_dim=7, measure_dim=5, k_neighbors=1, novelty_threshold=1, seed=100)

sols = rng.random((10, 7))
meas = 2 * rng.random((10, 5))
obj = rng.random((10))
ball_arch.add(sols, obj, meas)

stats_before = ball_arch.stats

where_new = rng.integers([10] * 5)[None]

n_sols = sols[where_new[0], ...]
n_meas = meas[where_new[0], ...]
n_obj = obj[where_new[0], ...]

ball_arch.add(n_sols, n_obj, n_meas)

stats_after_new = ball_arch.stats

n_obj_2 = n_obj + (rng.integers(-1, 2, n_obj.shape) * rng.random(n_obj.shape))

ball_arch.add(n_sols, n_obj_2, n_meas)

stats_after_new_improved = ball_arch.stats

c_sols = sols[[0], :]
c_meas = meas[[0], :]
c_obj = obj[[0]] + 1

ball_arch.add(c_sols, c_obj, c_meas)

stats_after_compete = ball_arch.stats
