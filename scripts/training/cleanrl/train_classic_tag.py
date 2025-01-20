# Problem
# Algo
# Population
# Island
# Archipelago


import time

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_tag_v3

# from algatross.algorithms.genetic.mo_aim.archipelago import DaskMOAIMArchipelago, RayMOAIMArchipelago
from algatross.agents.on_policy.ppo import TorchPPOAgent
from algatross.environments.mpe.simple_tag import train_island

# from algatross.algorithms.genetic.mo_aim.islands import RayMOAIMIslandUDI, RayMOAIMMainlandUDI

if __name__ == "__main__":
    seed = 1000
    island_env: ParallelEnv = simple_tag_v3.parallel_env(num_good=2, num_adversaries=3, num_obstacles=2)
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
            rollout_config={"gae_lambda": 0.85, "gamma": 0.85, "batch_size": 2500},
            # train_config={"device": "cuda:0", "sgd_minibatch_size": 1000, "num_sgd_iter": 10},
            train_config={"device": "cpu", "sgd_minibatch_size": 400, "num_sgd_iter": 20},
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
        cl_adv = []
        al = []
        al_adv = []
        r_m = []
        r_m_adv = []
        r_mx = []
        r_mx_adv = []
        r_mn = []
        r_mn_adv = []
        for agent_id, r in results.items():
            if "adversary" in agent_id:
                cl_adv.append(r["training_stats"]["critic_loss"].mean())
                al_adv.append(r["training_stats"]["actor_loss"].mean())
                r_m_adv.append(np.mean(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0))
                r_mx_adv.append(np.max(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=-np.inf))
                r_mn_adv.append(np.min(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=np.inf))
            else:
                cl.append(r["training_stats"]["critic_loss"].mean())
                al.append(r["training_stats"]["actor_loss"].mean())
                r_m.append(np.mean(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0))
                r_mx.append(np.max(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=-np.inf))
                r_mn.append(np.min(r["training_batch"]["rewards"], where=r["training_batch"]["rewards"] != 0, initial=np.inf))
        print(f"critic loss: allies={np.mean(cl)}, adversaries={np.mean(cl_adv)}")
        print(f"actor loss: allies={np.mean(al)}, adversaries={np.mean(al_adv)}")
        print(f"reward mean: allies={np.mean(r_m)}, adversaries={np.mean(r_m_adv)}")
        print(f"reward max: allies={np.mean(r_mx)}, adversaries={np.mean(r_mx_adv)}")
        print(f"reward min: allies={np.mean(r_mn)}, adversaries={np.mean(r_mn_adv)}")
        print()
    results  # noqa: B018
