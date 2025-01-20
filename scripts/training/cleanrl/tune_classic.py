# Problem
# Algo
# Population
# Island
# Archipelago


import numpy as np

from ray import tune

from pettingzoo import ParallelEnv  # noqa: TCH002
from pettingzoo.mpe import simple_spread_v3

# from algatross.algorithms.genetic.mo_aim.archipelago import DaskMOAIMArchipelago, RayMOAIMArchipelago
from algatross.agents.on_policy.ppo import TorchPPOAgent, train_island

# from algatross.algorithms.genetic.mo_aim.islands import RayMOAIMIslandUDI, RayMOAIMMainlandUDI


def train(config):
    seed = 1000
    island_env: ParallelEnv = simple_spread_v3.parallel_env(N=3)
    island_env.reset(seed=seed)
    eta = 0.8

    island_agent_config = {
        "critic_outs": 1,
        "shared_encoder": False,
        # "free_log_std": config["free_log_std"],
        "free_log_std": False,
        # "entropy_coeff": config["entropy_coeff"],
        "entropy_coeff": 0,
        # "kl_target": config["kl_target"],
        "kl_target": 0.2,
        # "kl_coeff": config["kl_coeff"],
        "kl_coeff": 0.2,
        "vf_coeff": 1.0,
        "logp_clip_param": config["logp_clip_param"],
        "vf_clip_param": None,
        "optimizer_kwargs": {"lr": config["lr"]},
    }

    agent_ids = island_env.possible_agents
    agent_map = {
        agent_id: PPOAgent(island_env.observation_space(agent_id), island_env.action_space(agent_id), **island_agent_config)
        for agent_id in agent_ids
    }
    cll = 0.0
    al1 = 0.0
    r_ml = 0.0
    r_mxl = 0.0
    r_mnl = 0.0
    for _i in range(1000):
        results = train_island(
            agent_map,
            island_env,
            trainable_agents=agent_ids,
            rollout_config={"gae_lambda": config["gae_lambda"], "gamma": config["gamma"], "batch_size": config["batch_size"]},
            train_config={"device": "cpu", "sgd_minibatch_size": config["sgd_minibatch_size"], "num_sgd_iter": config["num_sgd_iter"]},
        )
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
        cll = cll * eta + (1 - eta) * np.mean(cl)
        al1 = al1 * eta + (1 - eta) * np.mean(al)
        r_ml = r_ml * eta + (1 - eta) * np.mean(r_m)
        r_mxl = r_mxl * eta + (1 - eta) * np.mean(r_mx)
        r_mnl = r_mnl * eta + (1 - eta) * np.mean(r_mn)
    return {
        "critic_loss": cll,
        "actor_loss": al1,
        "mean_reward": r_ml,
        "max_reward": r_mxl,
        "min_reward": r_mnl,
    }


if __name__ == "__main__":
    search_space = {
        "logp_clip_param": tune.quniform(lower=0.05, upper=0.25, q=0.05),
        "lr": tune.sample_from(lambda: 10 ** tune.quniform(lower=-5, upper=-2.5, q=0.5).sample()),
        "gae_lambda": tune.quniform(lower=0.8, upper=0.95, q=0.05),
        "gamma": tune.quniform(lower=0.8, upper=0.95, q=0.05),
        "batch_size": tune.grid_search([1000, 5000, 10000]),
        "sgd_minibatch_size": tune.qrandint(lower=100, upper=1000, q=100),
        "num_sgd_iter": tune.grid_search([10, 20]),
    }
    tuner = tune.Tuner(train, param_space=search_space, tune_config=tune.TuneConfig(num_samples=10))
    results = tuner.fit()
    print(results.get_best_result(metric="mean_reward", mode="max").config)
    print(results.get_best_result(metric="max_reward", mode="max").config)
    print(results.get_best_result(metric="min_reward", mode="max").config)
    print(results.get_best_result(metric="actor_loss", mode="min").config)
    print(results.get_best_result(metric="critic_loss", mode="min").config)
