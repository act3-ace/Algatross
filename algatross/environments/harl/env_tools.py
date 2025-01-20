"""HARL environment utilities."""

import contextlib
import copy

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import numpy as np

import gymnasium

from harl.envs.env_wrappers import (
    CloudpickleWrapper,
    ShareDummyVecEnv as _ShareDummyVecEnv,
    ShareSubprocVecEnv as _ShareSubprocVecEnv,
    ShareVecEnv,
)

from algatross.utils.merge_dicts import merge_dicts


def shareworker(remote, parent_remote, env_fn_wrapper):
    """
    Get a customized shared worker which copies the original info after a call to reset.

    Parameters
    ----------
    remote : Connection
        A connection to a remote process
    parent_remote : Connection
        A connection to the parent process.
    env_fn_wrapper : Callable
        A wrapper around the environment.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(*data)
            if (isinstance(done, bool) and done) or np.all(done):
                original_info = copy.deepcopy(info)
                ob, s_ob, info, available_actions = env.reset()

                info[0]["original_obs"] = copy.deepcopy(ob)
                info[0]["original_state"] = copy.deepcopy(s_ob)
                info[0]["original_avail_actions"] = copy.deepcopy(available_actions)
                for idx, inf in enumerate(original_info):
                    info[idx]["original_info"] = copy.deepcopy(inf)

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, info, available_actions = env.reset()
            remote.send((ob, s_ob, info, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send(fr)
        elif cmd == "get_num_agents":
            remote.send(env.n_agents)
        else:
            raise NotImplementedError


class ShareDummyVecEnv(_ShareDummyVecEnv):
    """A wrapper around the ShareDummyVecEnv which uses a customized shared worker and reshapes the infos.

    Parameters
    ----------
    env_fns : Sequence[Callable]
        Functions for constructing copies of the environment.
    """

    envs: list
    """The list of environments in this dummy vectorized environment."""
    actions: gymnasium.spaces.Space | None
    """The actions to send to the environments."""
    n_agents: int
    """The number of agents in the environment."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None
        with contextlib.suppress(Exception):
            self.n_agents = env.n_agents

    def reset(self):  # noqa: D102
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, infos, available_actions = zip(*results, strict=True)
        return np.stack(obs), np.stack(share_obs), np.stack(infos), np.stack(available_actions)


class ShareSubprocVecEnv(_ShareSubprocVecEnv):
    """
    A wrapper around the ShareSubProcVecEnv which uses a customized shared worker and reshapes the infos.

    Parameters
    ----------
    env_fns : Sequence[Callable]
        Functions for constructing copies of the environment.
    spaces : Sequence[gymnasium.spaces.Space], optional
        The spaces used by the environment, default is :data:`python:None`
    """

    waiting: bool = False
    """Whether or not this wrapper is waiting on an asynchronous call."""
    closed: bool = False
    """Whether or not the subprocesses have been closed."""
    remotes: list[Connection]
    """Connections to the vector environments returned by :obj:`~multiprocessing.Pipe`."""
    work_remotes: list[Connection]
    """Connections to the worker environments returned by :obj:`~multiprocessing.Pipe`."""
    ps: list[Process]
    """The handles to the subrocesses."""
    n_agents: int
    """The number of agents in the environment."""

    def __init__(self, env_fns, spaces=None):  # noqa: ARG002
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)], strict=False)
        self.ps = [
            Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns, strict=False)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space, share_observation_space, action_space)

    def step_wait(self):  # noqa: D102
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results, strict=True)
        return (np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), np.stack(infos), np.stack(available_actions))

    def reset(self):  # noqa: D102
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, infos, available_actions = zip(*results, strict=True)
        return np.stack(obs), np.stack(share_obs), np.stack(infos), np.stack(available_actions)


def make_env(
    env_name: str,
    seed: int,
    n_threads: int,
    env_kwargs: dict,
    **kwargs,
) -> tuple[ShareDummyVecEnv | ShareSubprocVecEnv, list[str]]:
    """Make and return a vectorized version of the environment.

    Parameters
    ----------
    env_name : str
        Name of the environment to retrieve
    seed : int
        The base environment seed
    n_threads : int
        Number of threads to use
    env_kwargs : dict
        The keyword arguments to pass to the environment constructor

    Returns
    -------
    ShareDummyVecEnv | ShareSubprocVecEnv
        The vectorized environment
    list[str]
        The list of possible environment platforms

    Raises
    ------
    NotImplementedError
        If the ``env_name`` is not supported
    """
    if env_name == "pettingzoo_mpe":
        from algatross.environments.mpe.harl_env import PettingZooMPEEnv as BaseEnv  # noqa: PLC0415
    else:
        msg = "Can not support the " + env_name + "environment."
        raise NotImplementedError(msg)

    init_kwargs = copy.deepcopy(kwargs)
    init_kwargs = merge_dicts(env_kwargs, kwargs)
    env_base = BaseEnv(**init_kwargs)

    def get_env_fn(rank):
        def init_env():
            env = BaseEnv(**init_kwargs)
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)]), env_base.possible_platforms
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)]), env_base.possible_platforms


def make_eval_env(
    env_name: str,
    seed: int,
    n_threads: int,
    env_kwargs: dict,
    **kwargs,
) -> tuple[ShareDummyVecEnv | ShareSubprocVecEnv, list[str]]:
    """Make env for evaluation.

    Parameters
    ----------
    env_name : str
        Name of the environment to retrieve
    seed : int
        The base environment seed
    n_threads : int
        Number of threads to use
    env_kwargs : dict
        The keyword arguments to pass to the environment constructor

    Returns
    -------
    tuple[ShareDummyVecEnv | ShareSubprocVecEnv, list[str]]
        The vectorized environment and list of possible environment platforms

    Raises
    ------
    NotImplementedError
        If an unsupported environment is specified
    """
    if env_name == "dexhands":  # dexhands does not support running multiple instances
        msg = "dexhands does not support running multiple instances"
        raise NotImplementedError(msg)

    def get_env_fn(rank):
        def init_env():
            if env_name == "pettingzoo_mpe":
                from algatross.environments.mpe.harl_env import PettingZooMPEEnv  # noqa: PLC0415

                env = PettingZooMPEEnv(**kwargs, **env_kwargs)
            else:
                msg = "Can not support the " + env_name + "environment."
                raise NotImplementedError(msg)
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])
