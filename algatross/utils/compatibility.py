"""Compatibility with lower versions of ray."""

from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing


class Columns:
    """Columns object from later versions of ray 2."""

    OBS = SampleBatch.OBS
    INFOS = SampleBatch.INFOS
    ACTIONS = SampleBatch.ACTIONS
    ACTIONS_FOR_ENV = "actions_for_env"
    REWARDS = SampleBatch.REWARDS
    TERMINATEDS = SampleBatch.TERMINATEDS
    TRUNCATEDS = SampleBatch.TRUNCATEDS

    NEXT_OBS = SampleBatch.NEXT_OBS
    EPS_ID = SampleBatch.EPS_ID
    AGENT_ID = "agent_id"
    MODULE_ID = "module_id"

    SEQ_LENS = SampleBatch.SEQ_LENS
    T = SampleBatch.T
    STATE_IN = "state_in"
    STATE_OUT = "state_out"

    ACTION_DIST_INPUTS = SampleBatch.ACTION_DIST_INPUTS
    ACTION_PROB = SampleBatch.ACTION_PROB
    ACTION_LOGP = SampleBatch.ACTION_LOGP

    VF_PREDS = SampleBatch.VF_PREDS
    VALUES_BOOTSTRAPPED = SampleBatch.VALUES_BOOTSTRAPPED

    ADVANTAGES = Postprocessing.ADVANTAGES
    VALUE_TARGETS = Postprocessing.VALUE_TARGETS

    INTRINSIC_REWARDS = "intrinsic_rewards"

    LOSS_MASK = "loss_mask"
