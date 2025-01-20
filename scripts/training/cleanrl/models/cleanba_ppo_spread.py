import os
import queue
import random
import threading
import time
import uuid

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

import envpool
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tyro

from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from rich.pretty import pprint
from tensorboardX import SummaryWriter

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = Path(__file__).stem
    """The name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""
    track: bool = False
    # "if toggled, this experiment will be tracked with Weights and Biases"
    # wandb_project_name: str = "cleanRL"
    # "the wandb's project name"
    # wandb_entity: str = None
    # "the entity (team) of wandb's project"
    # capture_video: bool = False
    # "whether to capture videos of the agent performances (check out `videos` folder)"
    save_model: bool = False
    """Whether to save model into the `runs/{run_name}` folder."""
    upload_model: bool = False
    """Whether to upload the saved model to huggingface."""
    hf_entity: str = ""
    """The user or org name of the model repository from the Hugging Face Hub."""
    log_frequency: int = 10
    """The logging frequency of the model performance (in terms of `updates`)."""

    # Algorithm specific arguments
    # env_id: str = "Breakout-v5"
    # env_id: str = "SimpleSpreadContinuous-v0"
    env_id: str = "SimpleSpreadDiscrete-v0"
    """The id of the environment."""
    num_agents: int = 3
    """The number of agents acting in the environment."""
    total_timesteps: int = 50000000
    """Total timesteps of the experiments."""
    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer."""
    local_num_envs: int = 4
    # local_num_envs: int = 1
    "the number of parallel game environments"
    num_actor_threads: int = 2
    """The number of actor threads to use."""
    num_steps: int = 128
    """The number of steps to run in each environment per policy rollout."""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks."""
    gamma: float = 0.99
    """The discount factor gamma."""
    gae_lambda: float = 0.95
    """The lambda for the general advantage estimation."""
    num_minibatches: int = 4
    # num_minibatches: int = 1
    "the number of mini-batches"
    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps before performing an optimization step."""
    update_epochs: int = 4
    """The K epochs to update the policy."""
    norm_adv: bool = True
    """Toggles advantages normalization."""
    clip_coef: float = 0.1
    """The surrogate clipping coefficient."""
    ent_coef: float = 0.01
    """Coefficient of the entropy."""
    vf_coef: float = 0.5
    """Coefficient of the value function."""
    max_grad_norm: float = 0.5
    """The maximum norm for the gradient clipping."""
    channels: list[int] = field(default_factory=lambda: [16, 32, 32])
    """The channels of the CNN."""
    hiddens: list[int] = field(default_factory=lambda: [256])
    """The hiddens size of the MLP."""

    actor_device_ids: list[int] = field(default_factory=lambda: [0])
    """The device ids that actor workers will use."""
    learner_device_ids: list[int] = field(default_factory=lambda: [0])
    """The device ids that learner workers will use."""
    distributed: bool = False
    """Whether to use `jax.distirbuted`."""
    concurrency: bool = False
    """Whether to run the actor and learner concurrently."""

    # runtime arguments to be filled in
    local_batch_size: int = 0
    local_minibatch_size: int = 0
    num_updates: int = 0
    world_size: int = 0
    local_rank: int = 0
    num_envs: int = 0
    batch_size: int = 0
    minibatch_size: int = 0
    global_learner_devices: list[str] | None = None
    actor_devices: list[str] | None = None
    learner_devices: list[str] | None = None


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gymnasium",
            num_envs=num_envs,
            max_num_players=3,
            num_agents=3,
            num_landmarks=3,
            max_episode_steps=25,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.agents = [f"agent_{idx}" for idx in range(envs.config["num_agents"])]
        envs.num_agents = envs.config["num_agents"]
        envs.num_landmarks = envs.config["num_landmarks"]
        envs.landmarks = [f"landmark_{idx}" for idx in range(envs.config["num_landmarks"])]
        envs.is_vector_env = True
        return envs

    return thunk


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x  # noqa: RET504


class Network(nn.Module):
    channels: Sequence[int] = (16, 32, 32)
    hiddens: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x):
        # x = jnp.transpose(x, (0, 2, 3, 1))
        # x = x / (255.0)
        # for channels in self.channels:
        #     x = ConvSequence(channels)(x)
        # x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@flax.struct.dataclass
class FlaxAgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

    def __iter__(self):
        return iter(self.__dict__)


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logprobs: list
    values: list
    env_ids: list
    rewards: list
    truncations: list
    terminations: list
    firststeps: list  # first step of an episode


def rollout(  # noqa: PLR0914, PLR0915
    key: jax.random.PRNGKey,
    args: Args,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,  # noqa: ARG001
):
    envs = make_env(
        args.env_id,
        args.seed + jax.process_index() + device_thread_id,
        args.local_num_envs,
    )()
    len_actor_device_ids = len(args.actor_device_ids)
    global_step = 0
    start_time = time.time()

    @jax.jit
    def get_actions_and_values(
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        next_obs = jnp.array(next_obs)
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, next_obs)
        logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        # logits = Actor(envs.single_action_space.high[0][0] + 1).apply(params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = Critic().apply(params.critic_params, hidden)
        return next_obs, action, logprob, value.squeeze(), key

    # put data in the last index
    episode_returns = np.zeros((args.local_num_envs, envs.num_agents), dtype=np.float32)
    returned_episode_returns = np.zeros((args.local_num_envs, envs.num_agents), dtype=np.float32)
    episode_lengths = np.zeros((args.local_num_envs, envs.num_agents), dtype=np.float32)
    returned_episode_lengths = np.zeros((args.local_num_envs, envs.num_agents), dtype=np.float32)

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, info = envs.reset()
    next_done = jnp.zeros(args.local_num_envs, dtype=jax.numpy.bool_)

    @jax.jit
    def prepare_data(storage: list[Transition]) -> Transition:
        return jax.tree_map(lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage)

    for update in range(1, args.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if args.concurrency:
            if update != 2:  # noqa: PLR2004
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action_and_value` function that hangs until the params are ready.
                # This blocks the `get_action_and_value` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                # params.network_params["params"]["Dense_0"][
                params[0].network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        storage = []
        for _ in range(args.num_steps):
            cached_next_obs = next_obs.reshape(envs.num_envs, envs.num_agents, -1)
            cached_next_done = next_done
            global_step += len(next_done) * args.num_actor_threads * len_actor_device_ids * args.world_size
            inference_time_start = time.time()
            actions = []
            logprobs = []
            values = []
            # vmap_get_action_and_value = jax.vmap(get_action_and_value, in_axes=[0, 1, None], out_axes=[1, 1, 1, 1, None])
            # cached_next_obs, action, logprob, value, key = vmap_get_actions_and_values(params, cached_next_obs, key)
            for agent_idx in range(envs.num_agents):
                cached_next_obs[:, agent_idx, ...], action, logprob, value, key = get_actions_and_values(
                    params[agent_idx],
                    cached_next_obs[:, agent_idx, ...],
                    key,
                )
                actions.append(action)
                logprobs.append(logprob)
                values.append(value)
            action = jnp.stack(actions, axis=1)
            logprob = jnp.stack(logprobs, axis=1)
            value = jnp.stack(values, axis=1)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            # cpu_action = np.random.randint(0, envs.single_action_space.n, action.shape).reshape(-1, 1)
            # cpu_action = np.array(action.reshape(-1, 1))
            # cpu_action = np.array(action.reshape(-1, envs.num_agents, 1))
            # cpu_action = np.array(action).reshape(-1)
            cpu_action = np.array(action)
            # cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start

            env_send_time_start = time.time()
            env_id = info["env_id"]
            # for act in action.transpose(1, 0):
            #     next_obs, next_reward, next_done, next_trunc, info = envs.step(action={"action": act.reshape(-1)}, env_id=env_id)
            # next_obs, next_reward, next_done, next_trunc, info = envs.step(list(cpu_action), env_id=info["env_id"])
            # for _ in range(envs.num_envs):
            #     next_obs, next_reward, next_done, next_trunc, info = envs.step(cpu_action.reshape(-1), env_id=info["env_id"])
            next_obs, next_reward, next_done, _next_trunc, info = envs.step(cpu_action, env_id=info["env_id"])
            # next_obs, next_reward, next_done, next_trunc, info = envs.step(cpu_action)
            next_reward = next_reward.reshape(envs.num_envs, envs.num_agents, -1).sum(axis=-1)

            agent_terms = info["players"]["term"].reshape(envs.num_envs, envs.num_agents)
            agent_truncs = info["players"]["trunc"].reshape(envs.num_envs, envs.num_agents)
            env_id = info["env_id"]

            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_done,
                    actions=action,
                    logprobs=logprob,
                    values=value,
                    env_ids=env_id,
                    rewards=next_reward,
                    truncations=agent_truncs,
                    terminations=agent_terms,
                    firststeps=cached_next_done,
                ),
            )
            episode_returns[env_id] += next_reward
            returned_episode_returns[env_id] = np.where(
                agent_terms + agent_truncs,
                episode_returns[env_id],
                returned_episode_returns[env_id],
            )
            episode_returns[env_id] *= (1 - agent_terms) * (1 - agent_truncs)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                agent_terms + agent_truncs,
                episode_lengths[env_id],
                returned_episode_lengths[env_id],
            )
            episode_lengths[env_id] *= (1 - agent_terms) * (1 - agent_truncs)
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        avg_episodic_return = np.mean(returned_episode_returns)
        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *[jax.device_put_sharded(x, devices=learner_devices) for x in partitioned_storage],
        )
        # next_obs, next_done are still in the host
        sharded_next_obs = jax.device_put_sharded(
            np.split(
                np.stack(
                    [next_obs.reshape(envs.num_envs, envs.num_agents, -1)[:, agent_idx] for agent_idx in range(envs.num_agents)],
                    axis=1,
                ),
                len(learner_devices),
            ),
            devices=learner_devices,
        )
        sharded_next_done = jax.device_put_sharded(np.split(next_done, len(learner_devices)), devices=learner_devices)
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_next_obs,
            sharded_next_done,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        if update % args.log_frequency == 0:
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}",
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)
            writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/storage_time", storage_time, global_step)
            writer.add_scalar("stats/d2h_time", d2h_time, global_step)
            writer.add_scalar("stats/env_send_time", env_send_time, global_step)
            writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.num_actor_threads
                    * args.world_size
                    / (time.time() - update_time_start),
                ),
                global_step,
            )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    assert (  # noqa: S101
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (  # noqa: S101
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids) + len(args.actor_device_ids)),
        )
        print(list(range(len(args.learner_device_ids) + len(args.actor_device_ids))))

    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.minibatch_size = args.local_minibatch_size * args.world_size
    args.num_updates = args.total_timesteps // (args.local_batch_size * args.world_size)
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_devices", global_learner_devices)
    args.global_learner_devices = [str(item) for item in global_learner_devices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"
    # if args.track and args.local_rank == 0:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, *network_keys = jax.random.split(key, args.num_agents + 1)
    key, *actor_keys = jax.random.split(key, args.num_agents + 1)
    key, *critic_keys = jax.random.split(key, args.num_agents + 1)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    # env setup
    envs = make_env(args.env_id, args.seed, args.local_num_envs)()

    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    envs.reset()

    network = Network(args.channels, args.hiddens)
    actor = Actor(action_dim=envs.single_action_space.n)
    # actor = Actor(action_dim=envs.single_action_space.high[0][0] + 1)
    critic = Critic()
    network_params = [network.init(network_key, np.array([envs.single_observation_space.sample()])) for network_key in network_keys]
    agent_states = [
        TrainState.create(
            apply_fn=None,
            params=FlaxAgentParams(
                network_param,
                actor.init(actor_keys[agent_idx], network.apply(network_param, np.array([envs.single_observation_space.sample()]))),
                critic.init(critic_keys[agent_idx], network.apply(network_param, np.array([envs.single_observation_space.sample()]))),
            ),
            tx=optax.MultiSteps(
                optax.chain(
                    optax.clip_by_global_norm(args.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(
                        learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                        eps=1e-5,
                    ),
                ),
                every_k_schedule=args.gradient_accumulation_steps,
            ),
        )
        for agent_idx, network_param in enumerate(network_params)
    ]
    agent_states = flax.jax_utils.replicate(agent_states, devices=learner_devices)
    print(network.tabulate(network_keys[0], np.array([envs.single_observation_space.sample()])))
    print(actor.tabulate(actor_keys[0], network.apply(network_params[0], np.array([envs.single_observation_space.sample()]))))
    print(critic.tabulate(critic_keys[0], network.apply(network_params[0], np.array([envs.single_observation_space.sample()]))))

    @jax.jit
    def get_values(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
    ):
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, obs)
        value = Critic().apply(params.critic_params, hidden).squeeze(-1)
        return value  # noqa: RET504

    @jax.jit
    def get_logprob_entropy_value(
        params: flax.core.FrozenDict,
        obs: np.ndarray,
        actions: np.ndarray,
    ):
        hidden = Network(args.channels, args.hiddens).apply(params.network_params, obs)
        logits = Actor(envs.single_action_space.n).apply(params.actor_params, hidden)
        # logits = Actor(envs.single_action_space.high[0][0] + 1).apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(actions.shape[0]), actions]
        logits -= jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = Critic().apply(params.critic_params, hidden).squeeze(-1)
        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    # @jax.jit
    @partial(jax.jit, static_argnames=["agent_index"])
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Transition,
        agent_index: int,
    ):
        next_value = critic.apply(
            agent_state.params.critic_params,
            network.apply(agent_state.params.network_params, next_obs),
        ).squeeze()

        advantages = jnp.zeros_like(next_value)
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values[..., agent_index], next_value[None, :]], axis=0)  # noqa: PD011
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards[..., agent_index]),
            reverse=True,
        )
        return advantages, advantages + storage.values[..., agent_index]  # noqa: PD011

    def ppo_loss(params, obs, actions, behavior_logprobs, firststeps, advantages, target_values):  # noqa: ARG001
        newlogprob, entropy, newvalue = get_logprob_entropy_value(params=params, obs=obs, actions=actions)
        logratio = newlogprob - behavior_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - target_values) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    @jax.jit
    def single_device_update(
        agent_state: TrainState,
        sharded_storages: list,
        sharded_next_obs: list,
        sharded_next_done: list,
        key: jax.random.PRNGKey,
        agent_index: list,
    ):
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        next_obs = jnp.concatenate(sharded_next_obs)
        next_done = jnp.concatenate(sharded_next_done)
        ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
        advantages, target_values = compute_gae(agent_state, next_obs, next_done, storage, agent_index)
        if args.norm_adv:  # NOTE: per-minibatch advantages normalization
            advantages = advantages.reshape(advantages.shape[0], args.num_minibatches, -1)
            advantages = (advantages - advantages.mean((0, -1), keepdims=True)) / (advantages.std((0, -1), keepdims=True) + 1e-8)
            advantages = advantages.reshape(advantages.shape[0], -1)

        def update_epoch(carry, _):
            agent_state, key, agent_index = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches * args.gradient_accumulation_steps, -1) + x.shape[1:])
                return x  # noqa: RET504

            flatten_storage = jax.tree_map(flatten, storage)
            flatten_advantages = flatten(advantages)
            flatten_target_values = flatten(target_values)
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)
            shuffled_advantages = convert_data(flatten_advantages)
            shuffled_target_values = convert_data(flatten_target_values)

            def update_minibatch(agent_state, minibatch):
                mb_obs, mb_actions, mb_behavior_logprobs, mb_firststeps, mb_advantages, mb_target_values = minibatch
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    mb_obs,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_firststeps,
                    mb_advantages,
                    mb_target_values,
                )
                grads = jax.lax.pmean(grads, axis_name="local_devices")
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    shuffled_storage.obs[..., agent_index, :],
                    shuffled_storage.actions[..., agent_index],
                    shuffled_storage.logprobs[..., agent_index],
                    shuffled_storage.firststeps,
                    shuffled_advantages,
                    shuffled_target_values,
                ),
            )
            return (agent_state, key, agent_index), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (agent_state, key, agent_index), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            update_epoch,
            (agent_state, key, agent_index),
            (),
            length=args.update_epochs,
        )
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_devices,
    )

    params_queues = []
    rollout_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = flax.jax_utils.unreplicate([agent_state.params for agent_state in agent_states])
    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(args.num_actor_threads):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queues[-1],
                    params_queues[-1],
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * args.num_actor_threads + thread_id,
                    local_devices[d_id],
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []
        for d_idx, d_id in enumerate(args.actor_device_ids):  # noqa: B007
            for thread_id in range(args.num_actor_threads):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * args.num_actor_threads + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        loss = 0
        pg_loss = 0
        v_loss = 0
        entropy_loss = 0
        approx_kl = 0
        for agent_idx in range(envs.num_agents):
            (agent_states[agent_idx], ag_loss, ag_pg_loss, ag_v_loss, ag_entropy_loss, ag_approx_kl, learner_keys) = multi_device_update(
                agent_states[agent_idx],
                sharded_storages,
                [sno[..., agent_idx, :] for sno in sharded_next_obss],
                sharded_next_dones,
                learner_keys,
                jnp.array([agent_idx]),
            )
            loss += ag_loss
            pg_loss += ag_pg_loss
            v_loss += ag_v_loss
            entropy_loss += ag_entropy_loss
            approx_kl += ag_approx_kl

        unreplicated_params = flax.jax_utils.unreplicate([agent_state.params for agent_state in agent_states])
        for d_idx, d_id in enumerate(args.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(args.num_actor_threads):
                params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

        # record rewards for plotting purposes
        if learner_policy_version % args.log_frequency == 0:
            writer.add_scalar("stats/rollout_queue_get_time", np.mean(rollout_queue_get_time), global_step)
            writer.add_scalar(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
            writer.add_scalar("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
            writer.add_scalar("stats/params_queue_size", params_queues[-1].qsize(), global_step)
            print(
                global_step,
                (
                    f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, "
                    f"training time: {time.time() - training_time_start}s"
                ),
            )
            writer.add_scalar(
                "charts/learning_rate",
                # agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item(),
                np.mean([agent_state.opt_state[2][1].hyperparams["learning_rate"][-1].item() for agent_state in agent_states]),
                global_step,
            )
            writer.add_scalar("losses/value_loss", v_loss[-1].item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss[-1].item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss[-1].item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl[-1].item(), global_step)
            writer.add_scalar("losses/loss", loss[-1].item(), global_step)
        if learner_policy_version >= args.num_updates:
            break

    if args.save_model and args.local_rank == 0:
        if args.distributed:
            jax.distributed.shutdown()
        agent_states = flax.jax_utils.unreplicate(agent_states)
        for agent, agent_state in zip(envs.agents, agent_states, strict=True):
            model_path = f"runs/{run_name}/{args.exp_name}/{agent}.cleanrl_model"
            with open(model_path, "wb") as f:  # noqa: FURB103, PTH123
                f.write(
                    flax.serialization.to_bytes(
                        [
                            vars(args),
                            [
                                agent_state.params.network_params,
                                agent_state.params.actor_params,
                                agent_state.params.critic_params,
                            ],
                        ],
                    ),
                )
            print(f"model saved to {model_path}")
        # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        # with open(model_path, "wb") as f:
        #     f.write(
        #         flax.serialization.to_bytes(
        #             [
        #                 vars(args),
        #                 [
        #                     agent_state.params.network_params,
        #                     agent_state.params.actor_params,
        #                     agent_state.params.critic_params,
        #                 ],
        #             ],
        #         ),
        #     )
        # print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_envpool_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # if args.upload_model:
        #     from cleanrl_utils.huggingface import push_to_hub

        #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
        #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
        #     push_to_hub(
        #         args,
        #         episodic_returns,
        #         repo_id,
        #         "PPO",
        #         f"runs/{run_name}",
        #         f"videos/{run_name}-eval",
        #         extra_dependencies=["jax", "envpool", "atari"],
        #     )

    envs.close()
    writer.close()
