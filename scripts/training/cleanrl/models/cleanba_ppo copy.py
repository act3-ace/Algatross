import os  # noqa: N999
import uuid

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import envpool
import jax
import tyro

from rich.pretty import pprint

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
    env_id: str = "SimpleSpreadDiscrete-v0"
    """The id of the environment."""
    total_timesteps: int = 50000000
    """Total timesteps of the experiments."""
    learning_rate: float = 2.5e-4
    """The learning rate of the optimizer."""
    local_num_envs: int = 4
    """The number of parallel game environments."""
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
    """The number of mini-batches."""
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
    global_learner_decices: list[str] | None = None
    actor_devices: list[str] | None = None
    learner_devices: list[str] | None = None


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
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    print("global_learner_decices", global_learner_decices)
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    pprint(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"

    num_envs = 4
    num_players = 3

    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        max_num_players=num_players,
        num_agents=num_players,
        num_landmarks=3,
        seed=args.seed,
    )
    act_space = envs.action_space
    obs0, info = envs.reset()
    for _ in range(5000):
        if (_ + 1) % 250 == 0:
            print(f"iter {_}")
        # action = np.array([act_space.sample() for _ in range(args.local_num_envs)])
        action = np.array([act_space.sample() for _ in range(num_envs * num_players)])
        if (_ + 1) % 250 == 0:
            print(f"sending action {action} to environment")
        # obs0, rew0, terminated, truncated, info0 = envs.step(action[:, None], env_id=np.arange(1))
        obs0, rew0, terminated, truncated, info0 = envs.step(action.reshape(-1), env_id=np.arange(num_envs))
        if (_ + 1) % 250 == 0:
            print(f"reward {rew0.reshape(num_envs, -1).sum(-1)} from environment")
            print()
    envs.close()
