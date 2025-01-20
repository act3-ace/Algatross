# Problem
# Algo
# Population
# Island
# Archipelago


import time

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_spread_v3

# from algatross.algorithms.genetic.mo_aim.archipelago import DaskMOAIMArchipelago, RayMOAIMArchipelago
from algatross.agents.on_policy.ppo import TorchPPOAgent, train_island

# from algatross.algorithms.genetic.mo_aim.islands import RayMOAIMIslandUDI, RayMOAIMMainlandUDI

if __name__ == "__main__":
    seed = 1000
    island_env: ParallelEnv = simple_spread_v3.parallel_env(N=3)
    island_env.reset(seed=seed)

    island_agent_config = {
        "critic_outs": 1,
        "shared_encoder": False,
        "free_log_std": True,
        "entropy_coeff": 0,
        "kl_target": 0.2,
        "kl_coeff": 0.2,
        "vf_coeff": 1.0,
        "logp_clip_param": 0.2,
        "vf_clip_param": None,
        "optimizer_kwargs": {"lr": 3e-4},
    }

    agent_ids = island_env.possible_agents
    agent_map = {
        agent_id: TorchPPOAgent(island_env.observation_space(agent_id), island_env.action_space(agent_id), **island_agent_config)
        for agent_id in agent_ids
    }
    for i in range(1000):
        t0 = time.process_time_ns()
        results = train_island(
            agent_map,
            island_env,
            trainable_agents=agent_ids,
            rollout_config={"gae_lambda": 0.85, "gamma": 0.85, "batch_size": 5000},
            # train_config={"device": "cuda:0", "sgd_minibatch_size": 1000, "num_sgd_iter": 10},
            train_config={"device": "cpu", "sgd_minibatch_size": 400, "num_sgd_iter": 50},
        )
        t = time.process_time_ns() - t0
        print(f"iteration {i}")
        print(f"time: {t * 1e-9}")
        # if i % 100 == 0:
        #     for a, r in results.items():
        #         print(a, r["training_stats"])
        #         print()
        for agent in agent_map.values():
            agent.cpu()
        cl = []
        al = []
        r_m = []
        r_mx = []
        r_mn = []
        for r in results.values():
            cl.append(r["training_stats"]["critic_loss"].mean())
            al.append(r["training_stats"]["actor_loss"].mean())
            r_m.append(np.mean(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0))
            r_mx.append(np.max(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=-np.inf))
            r_mn.append(np.min(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=np.inf))
        print(f"critic loss: {np.mean(cl)}")
        print(f"actor loss: {np.mean(al)}")
        print(f"reward mean: {np.mean(r_m)}")
        print(f"reward max: {np.mean(r_mx)}")
        print(f"reward min: {np.mean(r_mn)}")
        print()
    results  # noqa: B018
