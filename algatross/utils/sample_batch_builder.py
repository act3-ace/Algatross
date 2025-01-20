"""A simple port of rllib's SampleBatchBuilder since the API is planned to be deprecated but is still useful for our purposes."""

import collections

from typing import Any

import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch

from algatross.utils.types import AgentID


def _to_float_array(v: list[Any]) -> np.ndarray:
    arr = np.array(v)
    if arr.dtype == np.float64:
        return arr.astype(np.float32)  # save some memory
    return arr


class SampleBatchBuilder:
    """Util to build a SampleBatch incrementally.

    For efficiency, SampleBatches hold values in column form (as arrays).
    However, it is useful to add data one row (dict) at a time.
    """

    _next_unroll_id = 0  # disambiguates unrolls within a single episode

    def __init__(self):
        self.buffers: dict[str, list] = collections.defaultdict(list)
        self.count = 0

    def add_values(self, **values: dict) -> None:
        """
        Add the given dictionary (row) of values to this batch.

        Parameters
        ----------
        `**values` : dict
            Additional values to add to each batch.
        """
        for k, v in values.items():
            self.buffers[k].append(v)
        self.count += 1

    def add_batch(self, batch: SampleBatch) -> None:
        """
        Add the given batch of values to this batch.

        Parameters
        ----------
        batch : SampleBatch
            The samplebatch to concatenate to the builders batches.
        """
        for k, column in batch.items():
            self.buffers[k].extend(column)
        self.count += batch.count

    def build_and_reset(self) -> SampleBatch:
        """Return a sample batch including all previously added values.

        Returns
        -------
        SampleBatch
            The constructed sample batch.
        """
        batch = SampleBatch({k: _to_float_array(v) for k, v in self.buffers.items()})
        if SampleBatch.UNROLL_ID not in batch:
            batch[SampleBatch.UNROLL_ID] = np.repeat(SampleBatchBuilder._next_unroll_id, batch.count)
            SampleBatchBuilder._next_unroll_id += 1
        self.buffers.clear()
        self.count = 0
        return batch


def concat_agent_buffers(rollout_buffers: list[dict[AgentID, SampleBatchBuilder]]) -> dict[AgentID, SampleBatchBuilder]:
    """Concatenate agent buffers from parallel rollouts.

    Parameters
    ----------
    rollout_buffers : list[dict[AgentID, SampleBatchBuilder]]
        The list of rollout buffers to concatenate together.

    Returns
    -------
    dict[AgentID, SampleBatchBuilder]
        The concatenated buffers.
    """
    if len(rollout_buffers) == 1:
        return rollout_buffers[0]
    rb = rollout_buffers[0]
    for buffer in rollout_buffers[1:]:
        for agent_id, agent_buffer in buffer.items():
            rb[agent_id].add_batch(agent_buffer)  # type: ignore[arg-type]
    return rb
