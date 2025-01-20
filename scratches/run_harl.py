from algatross.agents.on_policy.harl.happo import TorchHAPPOAgent
from algatross.configs.harl.runners import HARLRunnerConfig
from algatross.environments.harl.runners.on_policy_ha import OnPolicyHARunner
from algatross.utils.types import AgentID, ConstructorData
import numpy as np

seed = 1000
n_rollout_threads = 20
episode_length = 200
env_name = "pettingzoo_mpe"
scenario = "simple_spread_v3"

agent_constructors: dict[AgentID, ConstructorData] = {"controller_0": ConstructorData(constructor=TorchHAPPOAgent)}
runner_config = HARLRunnerConfig()
runner = OnPolicyHARunner(
    agent_constructors=agent_constructors,
    platform_map={"controller_0": ["agent_0", "agent_1", "agent_2"]},
    env_name=env_name,
    continuous_actions=True,
    n_rollout_threads=n_rollout_threads,
    scenario=scenario,
    seed=seed,
)
eval_results = {}
results = {}
for i in range(10):
    eval_results[i] = runner(
        train=False,
        batch_size=10_000,
        trainable_agents=["agent_0", "agent_1", "agent_2"],
        reward_metrics={
            "agent_0": ["original_rewards"],
            "agent_1": ["original_rewards"],
            "agent_2": ["original_rewards"],
        },
    )
    results[i] = runner(
        train=True,
        batch_size=1_000_000,
        trainable_agents=["agent_0", "agent_1", "agent_2"],
        reward_metrics={
            "agent_0": ["original_rewards"],
            "agent_1": ["original_rewards"],
            "agent_2": ["original_rewards"],
        },
    )
eval_results[i + 1] = runner(
    train=False,
    batch_size=10_000,
    trainable_agents=["agent_0", "agent_1", "agent_2"],
    reward_metrics={
        "agent_0": ["original_rewards"],
        "agent_1": ["original_rewards"],
        "agent_2": ["original_rewards"],
    },
)
result_rewards = [
    np.mean(
        [
            result[agent_id]["extra_info"]["rollout_buffer"]["rewards"].reshape(-1, 25).sum(axis=-1)
            for agent_id in ["agent_0", "agent_1", "agent_2"]
        ]
    )
    for result in results.values()
]
eval_rewards = [
    np.mean(
        [
            eval_result[agent_id]["extra_info"]["rollout_buffer"]["rewards"].reshape(-1, 25).sum(axis=-1)
            for agent_id in ["agent_0", "agent_1", "agent_2"]
        ]
    )
    for eval_result in eval_results.values()
]
result_msg = [f"{ep} returns: {rew}\n" for ep, rew in enumerate(result_rewards)]
result_msg = "\tEpisode ".join(["", *result_msg])
result_msg = "-" * 50 + "\nTrain Results\n" + result_msg

eval_msg = [f"{ep} returns: {rew}\n" for ep, rew in enumerate(eval_rewards)]
eval_msg = "\tEpisode ".join(["", *eval_msg])
eval_msg = "\n" + "-" * 50 + "\nEvaluation Results\n" + eval_msg
print(result_msg)
print(eval_msg)
print(results)
