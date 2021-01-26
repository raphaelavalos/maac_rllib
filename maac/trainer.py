import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ray.rllib import SampleBatch
from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution import ParallelRollouts, StoreToReplayBuffer, Replay, TrainOneStep, UpdateTargetNetwork, \
    Concurrently, StandardMetricsReporting
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator, _NextValueNotReady

from maac import MAACTorchModel, MAACTorchPolicy

logger = logging.getLogger(__name__)

DEFAULT_POLICY = "default_policy"


# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Model ===


    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of iterations.
    "timesteps_per_iteration": 1000,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 1,
    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    "horizon": 100,
    "no_done_at_end": True,
    "replay_sequence_length": 1,
    "gamma": 0.99,
    "num_envs_per_worker": 12,

    # === Optimization ===
    # Learning rate for adam optimizer
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    # Learning rate schedule
    "lr_schedule": None,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    "grad_clip": 40,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1024,
    # Update the replay buffer with this many samples at once. Note that
    # this setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 8,
    "batch_mode": "truncate_episodes",
    "train_batch_size": 1024,
    "log_level": 'DEBUG',

    "actor_hiddens": [128, 128],
    "nbr_heads": 4,
    "tau": .005,
    "critic_emb_dim": 128,
    "batch_norm": True,
    "reguralize_policy": True,
    "huber_loss": False,

    "lr_actor": 1e-3,
    "lr_schedule_q_network": None,
    "lr_critic": 1e-3,
    "lr_schedule_critic": None,
    "communication": False,
    "recurrent": False,
    # "differentiable_communication": True,
    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    "num_gpus_per_worker": 0.15,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
    # "evaluation_interval": 5,
    # "custom_eval_function": evaluate,
    "round_robin_weights": [1, 4],
    # "mean_communication": True,
    # "rescale_rewards": None,
    # "single_gate": True,True
})

def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the Simple Q algorithm. Defines the distributed dataflow.

    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: A local iterator over training metrics.
    """
    local_replay_buffer = LocalReplayBuffer(
        num_shards=1,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        replay_mode=config["multiagent"]["replay_mode"],
        replay_sequence_length=config["replay_sequence_length"])

    rollouts = ParallelRollouts(workers, mode="bulk_sync")

    # (1) Generate rollouts and store them in our local replay buffer.
    store_op = rollouts.for_each(
        StoreToReplayBuffer(local_buffer=local_replay_buffer))

    # (2) Read and train on experiences from the replay buffer.
    replay_op = Replay(local_buffer=local_replay_buffer) \
        .for_each(TrainOneStep(workers)) \
        .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"]))

    # Alternate deterministically between (1) and (2).
    train_op = Concurrently(
        [store_op, replay_op], mode="round_robin",
        round_robin_weights=[1, 4], output_indexes=[1])

    return StandardMetricsReporting(train_op, workers, config)

MAACTrainer = build_trainer(
    name="MAACTrainer",
    default_policy=MAACTorchPolicy,
    default_config=DEFAULT_CONFIG,
    execution_plan=execution_plan,
)