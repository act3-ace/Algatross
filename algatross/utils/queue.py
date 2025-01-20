"""Queue utilities."""

import asyncio

from collections.abc import Collection, Hashable, Iterable
from typing import Any

import ray

from ray.util.queue import Empty, Full, _QueueActor  # noqa: PLC2701


class MultiQueue:
    """
    Implementation of a single queue actor which holds multiple asynchronous queues.

    Parameters
    ----------
    queue_keys : Iterable[Hashable]
        The keys under which to store items. Each key will contain its own :class:`~asyncio.Queue`
    maxsize : int, optional
        The maximum size for all of the queues, default is 0.
    actor_options : dict | None
        Options to pass to the actor constructor, default is :data:`python:None`.
    """

    def __init__(self, queue_keys: Iterable[Hashable], maxsize: int = 0, actor_options: dict | None = None) -> None:
        actor_options = actor_options or {}
        self.maxsize = maxsize
        self.queue_keys = queue_keys
        self.actor = ray.remote(_MultiQueueActor).options(**actor_options).remote(self.maxsize, self.queue_keys)  # type: ignore[attr-defined]

    def __len__(self) -> int:  # noqa: D105
        return self.size()

    def size(self, *, key: Hashable | None = None) -> int:
        """
        Get size of the queue.

        Parameters
        ----------
        key : Hashable | None, optional
            The queue whose size should be retrieved, :data:`python:None`.

        Returns
        -------
        int
            The size of the queue.
        """
        return ray.get(self.actor.qsize.remote(key=key))

    def qsize(self, *, key: Hashable | None = None) -> int:
        """
        Get size of the queue.

        Parameters
        ----------
        key : Hashable | None, optional
            The queue to fetch, :data:`python:None`.

        Returns
        -------
        int
            The size of the queue.
        """
        return self.size(key=key)

    def empty(self, *, key: Hashable | None = None) -> bool:
        """Whether the queue is empty.

        Parameters
        ----------
        key : Hashable | None, optional
            The queue to fetch, :data:`python:None`.

        Returns
        -------
        bool
            Whether the queue is empty.
        """
        return ray.get(self.actor.empty.remote(key=key))

    def full(self, *, key: Hashable | None = None) -> bool:
        """Whether the queue is full.

        Parameters
        ----------
        key : Hashable | None, optional
            The queue to fetch, :data:`python:None`.

        Returns
        -------
        bool
            Whether the queue is full.
        """
        return ray.get(self.actor.full.remote(key=key))

    def put(
        self,
        item: Any,  # noqa: ANN401
        block: bool = True,
        timeout: float | None = None,
        *,
        key: Hashable | None = None,
    ) -> None:
        """Add an item to the queue.

        If block is True and the queue is full, blocks until the queue is no
        longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Parameters
        ----------
        item : Any
            The time to be put into the :class:`~asyncio.Queue`.
        block : bool, optional
            Whether to block the execution in the main thread, default is :data:`python:True`.
        timeout : float | None, optional
            The length of time to wait for the resource to be free before erroring, :data:`python:None`.
        key : Hashable | None, optional
            The queue into which the item will be put, :data:`python:None`.

        Raises
        ------
        Full
            If a value is put but the queue is full.
        """
        if not block:
            try:
                ray.get(self.actor.put_nowait.remote(item, key=key))
            except asyncio.QueueFull as err:
                raise Full from err
        else:
            ray.get(self.actor.put.remote(item, key=key), timeout=timeout)

    async def put_async(
        self,
        item: Any,  # noqa: ANN401
        block: bool = True,
        *,
        key: Hashable | None = None,
    ) -> None:
        """
        Add an item to the queue.

        If block is True and the queue is full,
        blocks until the queue is no longer full or until timeout.

        There is no guarantee of order if multiple producers put to the same
        full queue.

        Parameters
        ----------
        item : Any
            The item to put into the :class:`~asyncio.Queue`.
        block : bool
            Whether to block the calling thread.
        key : Hashable | None, optional
            The name of the queue to retrieve, :data:`python:None`.

        Raises
        ------
        Full
            If the queue is full.
        """
        if not block:
            try:
                await self.actor.put_nowait.remote(item, key=key)
            except asyncio.QueueFull as err:
                raise Full from err
        else:
            await self.actor.put.remote(item, key=key)

    def get(self, block: bool = True, timeout: float | None = None, *, key: Hashable | None = None) -> Any:  # noqa: ANN401
        """Get an item from the queue.

        If block is True and the queue is empty, blocks until the queue is no
        longer empty or until timeout.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Parameters
        ----------
        block : bool, optional
            Whether to block execution in the calling thread, :data:`python:True`.
        timeout : float | None, optional
            The length of time to wait before raising a timeout error, :data:`python:None`.
        key : Hashable | None, optional
            The queue to get, :data:`python:None`.

        Returns
        -------
        Any
            The contentes of the queue.

        Raises
        ------
        Empty
            If the queue is empty.
        ValueError
            If timeout is negative.
        """
        if not block:
            try:
                return ray.get(self.actor.get_nowait.remote(key=key))
            except asyncio.QueueEmpty as err:
                raise Empty from err
        elif timeout is not None and timeout < 0:
            msg = "'timeout' must be a non-negative number"
            raise ValueError(msg)
        else:
            return ray.get(self.actor.get.remote(timeout, key=key))

    async def get_async(self, block: bool = True, *, key: Hashable | None = None) -> Any:  # noqa: ANN401
        """Get an item from the queue.

        There is no guarantee of order if multiple consumers get from the
        same empty queue.

        Parameters
        ----------
        block : bool, optional
            Whether to block execution in the calling thread, :data:`python:True`.
        key : Hashable | None, optional
            The queue to retrieve, :data:`python:None`.

        Returns
        -------
        Any
            The contents of the queue.

        Raises
        ------
        Empty
            If the queue is empty.
        """
        if not block:
            try:
                return await self.actor.get_nowait.remote(key=key)
            except asyncio.QueueEmpty as err:
                raise Empty from err
        else:
            return await self.actor.get.remote(key=key)

    def put_nowait(self, item: Any, *, key: Hashable | None = None) -> None:  # noqa: ANN401
        """Equivalent to put(item, block=False).

        Parameters
        ----------
        item : Any
            The item to put in the queue.
        key : Hashable | None
            The name of the queue to put the item into.
        """
        self.put(item, block=False, key=key)

    def put_nowait_batch(self, items: Iterable, *, key: Hashable | None = None) -> None:
        """Take in a list of items and puts them into the queue in order.

        Parameters
        ----------
        items : Iterable
            The items to put into the queue.
        key : Hashable | None, optional
            The name of the queue to update, :data:`python:None`.

        Raises
        ------
        TypeError
            If ``items`` is not :class:`~typing.Iterable`.
        """
        if not isinstance(items, Iterable):
            msg = "Argument 'items' must be an Iterable"
            raise TypeError(msg)

        ray.get(self.actor.put_nowait_batch.remote(items, key=key))

    def get_nowait(self, *, key: Hashable | None = None) -> dict[Hashable, Any]:
        """Equivalent to get(block=False).

        Parameters
        ----------
        key : Hashable | None, optional
            The queue to get, default is :data:`python:None`.

        Returns
        -------
        dict[Hashable, Any]
            The dictionary containing the key and the :class:`~asyncio.Queue` entries.
        """
        return self.get(block=False, key=key)

    def get_nowait_batch(self, num_items: int, *, key: Hashable | None = None) -> list[Any]:
        """Get items from the queue and returns them in a list in order.

        Parameters
        ----------
        num_items : int
            Number of items to retrieve from the queue.
        key : Hashable | None, optional
            The queue to retrieve, default is :data:`python:None`.

        Returns
        -------
        list[Any]
            The fetched results from the :class:`~asyncio.Queue` contained in ``key``.

        Raises
        ------
        TypeError
            If ``num_items`` is not an integer.
        ValueError
            If ``num_items`` is negative.
        """
        if not isinstance(num_items, int):
            msg = "Argument 'num_items' must be an int"
            raise TypeError(msg)
        if num_items < 0:
            msg = "'num_items' must be nonnegative"
            raise ValueError(msg)

        return ray.get(self.actor.get_nowait_batch.remote(num_items, key=key))

    def shutdown(self, force: bool = False, grace_period_s: int = 5) -> None:
        """Terminate the underlying QueueActor.

        All of the resources reserved by the queue will be released.

        Parameters
        ----------
        force : bool, optional
            If True, forcefully kill the actor, causing an immediate failure. If False, graceful
            actor termination will be attempted first, before falling back to a forceful kill. Default is False.
        grace_period_s : int, optional
            If force is False, how long in seconds to wait for graceful termination before falling back to
            forceful kill. Default is 5.
        """
        if self.actor:
            if force:
                ray.kill(self.actor, no_restart=True)
            else:
                done_ref = self.actor.__ray_terminate__.remote()
                _done, not_done = ray.wait([done_ref], timeout=grace_period_s)
                if not_done:
                    ray.kill(self.actor, no_restart=True)
        self.actor = None


class _MultiQueueActor(_QueueActor):
    def __init__(self, maxsize, queue_keys):
        self.maxsize = maxsize
        self.queues: dict[Hashable, asyncio.Queue] = {key: asyncio.Queue(self.maxsize) for key in queue_keys}
        self._actor_name = ray.get_runtime_context().get_actor_name()

    def qsize(self, key: Hashable | None = None) -> int:
        if key is None:
            return sum(x.qsize() for x in self.queues.values())
        return self.queues[key].qsize()

    def empty(self, *, key: Hashable | None = None) -> bool:
        if key is None:
            return all(x.empty() for x in self.queues.values())
        return self.queues[key].empty()

    def full(self, *, key: Hashable | None = None) -> bool:
        if key is None:
            return all(x.full() for x in self.queues.values())
        return self.queues[key].full()

    async def put(self, item: Any, *, key: Hashable | None = None):  # type: ignore[override] # noqa: ANN401
        if key is None:
            for k in self.queues:
                await asyncio.wait_for(self.queues[k].put(item), timeout=None)
        else:
            await asyncio.wait_for(self.queues[key].put(item), timeout=None)

    async def get(self, *, key: Hashable | None = None) -> dict[Hashable, Any]:  # type: ignore[override]
        if key is None:
            return {k: await asyncio.wait_for(self.queues[k].get(), timeout=None) for k in self.queues}
        return {key: await asyncio.wait_for(self.queues[key].get(), timeout=None)}

    def put_nowait(self, item: Any, *, key: Hashable | None = None):  # noqa: ANN401
        if key is None:
            dict(self.foreach_queue("put_nowait", item))
        else:
            self.queues[key].put_nowait(item)

    def put_nowait_batch(self, items: Collection, *, key: Hashable | None = None):
        # If maxsize is 0, queue is unbounded, so no need to check size.
        if self.maxsize > 0 and len(items) + self.qsize(key=key) > self.maxsize:
            msg = f"Cannot add {len(items)} items to queue of size {self.qsize()} and maxsize {self.maxsize}."
            raise Full(msg)
        if key is None:
            for item in items:
                dict(self.foreach_queue("put_nowait", item))
        else:
            for item in items:
                self.queues[key].put_nowait(item)

    def get_nowait(self, *, key: Hashable | None = None) -> dict[Hashable, Any]:
        if key is None:
            return dict(self.foreach_queue("get_nowait"))
        return {key: self.queues[key].get_nowait()}

    def get_nowait_batch(self, num_items: int, *, key: Hashable | None = None) -> dict[Hashable, Any]:
        if num_items > self.qsize():
            msg = f"Cannot get {num_items} items from queue of size {self.qsize()}."
            raise Empty(msg)
        if key is None:
            return {k: [self.queues[k].get_nowait() for _ in range(num_items)] for k in self.queues}
        return {key: [self.queues[key].get_nowait() for _ in range(num_items)]}

    def foreach_queue(self, fn_str, *args, **kwargs):
        for key in self.queues:
            yield (key, getattr(self.queues[key], fn_str)(*args, **kwargs))

    def __repr__(self) -> str:
        return f"{self._actor_name} <{self.__class__.__name__}>" if self._actor_name else self.__class__.__name__
