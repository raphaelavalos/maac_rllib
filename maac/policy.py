from typing import Tuple

import gym
import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper, \
    TorchMultiCategorical
from ray.rllib.policy import build_torch_policy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict

import maac
from maac.model import MAACTorchModel

torch, nn = try_import_torch()
F = nn.functional


def make_model_and_action_dist(
        policy: Policy,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, TorchDistributionWrapper]:
    keys = [
        "actor_hiddens",
        "batch_norm",
        "critic_emb_dim",
        "nbr_heads"
    ]

    model_config = {**config["model_config"],
                    **{key: config[key] for key in keys}}
    policy.model = MAACTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        model_config=model_config,
        num_outputs=sum(action_space.nvec)
    )
    policy.target_model = MAACTorchModel(
        obs_space=obs_space,
        action_space=action_space,
        model_config=model_config,
        num_outputs=sum(action_space.nvec)
    )
    policy.target_model.load_state_dict(policy.model.state_dict())

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    policy.target_model.to(device)

    class MAACCOMTorchMultiCategorical(TorchMultiCategorical):
        def __init__(self, input, model):
            super().__init__(input, model, action_space.nvec)

    return policy.model, MAACCOMTorchMultiCategorical


def make_optimizers(policy: Policy,
                    config: TrainerConfigDict):
    policy._critic_optimizer = torch.optim.Adam(
        policy.model.critic_variables(),
        lr=config["critic_lr"],
        eps=config["adam_epsilon"]
    )

    policy._actor_optimizers = [
        torch.optim.Adam(
            policy.model.actor_variable_per_agent(i),
            lr=config["actor_lr"],
            eps=config["adam_epsilon"],
        )
        for i in range(policy.model.nbr_agents)
    ]

    return policy._actor_optimizers + [policy._critic_optimizer]


class TargetNetworkMixin:
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, config: TrainerConfigDict):
        # Hard initial update from Q-net(s) to target Q-net(s).
        self.update_target(tau=1.0)

    def update_target(self, tau=None):
        tau = tau or self.config.get("tau")
        # Update_target_fn will be called periodically to copy Q network to
        # target Q network, using (soft) tau-synching.
        # Full sync from Q-model to target Q-model.
        if tau == 1.0:
            self.target_model.load_state_dict(
                self.model.state_dict())
        # Partial (soft) sync using tau-synching.
        else:
            model_vars = list(self.model.parameters())
            target_model_vars = list(self.target_model.parameters())
            assert len(model_vars) == len(target_model_vars), \
                (model_vars, target_model_vars)
            for var, var_target in zip(model_vars, target_model_vars):
                var_target.data = tau * var.data + \
                                  (1.0 - tau) * var_target.data


def loss_policy(policy: Policy, model: ModelV2,
                dist_class: TorchDistributionWrapper, sample_batch: SampleBatch,
                alpha: float):
    B = sample_batch[SampleBatch.OBS].shape[0]
    logits_t, _ = model(sample_batch)
    logits_t = logits_t.view(B, model.nbr_agents, model.nbr_actions)
    log_prob_t = F.log_softmax(logits_t, dim=-1)
    prob_t = torch.exp(log_prob_t)
    act_t = prob_t.argmax(dim=-1, keepdim=True)

    q_t, _, _ = model.get_q_values({
        SampleBatch.OBS: sample_batch[SampleBatch.OBS],
        SampleBatch.ACTIONS: act_t.squeeze(-1)
    })

    log_prob_t_selected = torch.gather(log_prob_t, -1, act_t.long()).squeeze(-1)
    q_t_selected = torch.gather(q_t, -1, act_t.long()).squeeze(-1)
    policy.actor_loss = - log_prob_t_selected * (
                - alpha * log_prob_t_selected + (
                    q_t_selected - (prob_t * q_t).sum(-1))).detach()
    if policy.config["reguralize_policy"]:
        policy.policy_reg = torch.square(logits_t).mean(-1)

    policy.entropy = - (log_prob_t * prob_t).sum(-1).mean(0)

    policy.actor_loss = policy.actor_loss.mean(0)
    policy.policy_reg = policy.policy_reg.mean(0)

    losses = []
    if not model.parameter_sharing:
        for actor_loss, reg_loss in zip(
                policy.actor_loss.split(1, dim=0),
                policy.policy_reg.split(1, dim=0)):
            losses.append(actor_loss + 1e-3 * reg_loss)
    else:
        losses.append(
            policy.actor_loss.mean() + 1e-3 * policy.policy_reg.mean())

    return losses


def loss_critic(policy: Policy, model: ModelV2,
                dist_class: TorchDistributionWrapper, sample_batch: SampleBatch,
                alpha: float):
    gamma = policy.config["gamma"]
    B = sample_batch[SampleBatch.OBS].shape[0]
    sample_batch_tp1 = {SampleBatch.OBS: sample_batch[SampleBatch.NEXT_OBS]}

    target_logits_tp1, _ = policy.target_model(sample_batch_tp1)
    target_logits_tp1 = target_logits_tp1.view(B, model.nbr_agents,
                                               model.nbr_actions)
    target_log_prob_tp1 = F.log_softmax(target_logits_tp1, dim=-1)
    target_prob_tp1 = torch.exp(target_log_prob_tp1)
    target_act_tp1 = target_prob_tp1.argmax(dim=-1, keepdim=True)
    target_log_prob_tp1_selected = target_log_prob_tp1.gather(-1,
                                                              target_act_tp1).squeeze(
        -1)

    q_t, q_t_reg, q_t_entropy = model.get_q_values({
        SampleBatch.OBS: sample_batch[SampleBatch.OBS],
        SampleBatch.ACTIONS: sample_batch[SampleBatch.ACTIONS]
    })

    q_t_selected = torch.gather(
        q_t, -1, sample_batch[SampleBatch.ACTIONS].long().unsqueeze(-1)) \
        .squeeze(-1)

    target_q_tp1, _, _ = policy.target_model.get_q_values({
        SampleBatch.OBS: sample_batch[SampleBatch.NEXT_OBS],
        SampleBatch.ACTIONS: target_act_tp1.squeeze(-1),
    })

    target_q_tp1_selected = target_q_tp1.gather(-1, target_act_tp1).squeeze(-1)

    # Critic loss

    reward = sample_batch["real_reward"]
    done = torch.logical_not(
        sample_batch[SampleBatch.DONES].unsqueeze(-1)).float()
    y = reward + done * gamma * (target_q_tp1_selected
                                 - alpha * target_log_prob_tp1_selected)

    y = y.detach()
    td_error = q_t_selected - y
    policy.critic_loss = torch.pow(td_error, 2.0).mean(0).sum()
    policy.critic_reg = q_t_reg.mean(0).sum()
    policy.critic_entropy = q_t_entropy.mean(0)
    return policy.critic_loss + policy.critic_reg


def loss_fn(policy: Policy, model: ModelV2,
            dist_class: TorchDistributionWrapper, sample_batch: SampleBatch):
    alpha = .01
    actor_loss = loss_policy(policy, model, dist_class, sample_batch, alpha)
    critic_loss = loss_critic(policy, model, dist_class, sample_batch, alpha)
    return actor_loss + [critic_loss]


def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before SimpleQTorchPolicy
    initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)
    policy.view_requirements.update({
        SampleBatch.INFOS: ViewRequirement(SampleBatch.INFOS,
                                           used_for_training=True),
        "real_reward": ViewRequirement(
            space=gym.spaces.Box(float('-inf'), float('inf'),
                                 (policy.model.nbr_agents,)))
    })


def stats(policy, train_batch):
    stat = policy.get_exploration_info().copy()
    assert not policy.model.parameter_sharing
    for i in range(policy.model.nbr_agents):
        stat["agent_{}_actor_loss".format(i)] = policy.actor_loss[i]
    for i in range(policy.model.nbr_agents):
        stat["agent_{}_actor_reg".format(i)] = policy.policy_reg[i]
    for i in range(policy.model.nbr_agents):
        stat["agent_{}_entropy".format(i)] = policy.entropy[i]
    for i in range(policy.model.nbr_agents):
        stat["agent_{}_critic_entropy".format(i)] = {
            "head_{}".format(j): policy.critic_entropy[i, j]
            for j in range(policy.model.num_heads)
        }

    stat["critic_loss"] = policy.critic_loss
    stat["critic_reg"] = policy.critic_reg

    return stat


def apply_grad_clipping(policy, optimizer, loss):
    info = {}
    if optimizer is policy._critic_optimizer:
        grad_clip = 10 * policy.model.nbr_agents
        for p in policy.model.get_attention_parameters():
            p.grad.data.mul_(1. / policy.model.nbr_agents)
        label = "critic"
    else:
        grad_clip = .5
        label = "agent_{}".format(policy._actor_optimizers.index(optimizer))

    if True:
        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(
                filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                grad_gnorm = nn.utils.clip_grad_norm_(
                    params, grad_clip)
                if isinstance(grad_gnorm, torch.Tensor):
                    grad_gnorm = grad_gnorm.cpu().numpy()
                info[label + "_grad"] = grad_gnorm.item()
    return info


def postprocess_trajectory(policy, sample_batch,
                           other_agent_batches=None,
                           episode=None):
    if hasattr(policy.model.obs_space,
               'original_space') and "prev_reward" in policy.model.obs_space.original_space.spaces:
        sample_batch["real_reward"] = \
        restore_original_dimensions(sample_batch[SampleBatch.NEXT_OBS],
                                    policy.model.obs_space, np)['prev_reward']
    return sample_batch


MAACTorchPolicy = build_torch_policy(
    name="MAACTorchPolicy",
    make_model_and_action_dist=make_model_and_action_dist,
    optimizer_fn=make_optimizers,
    get_default_config=lambda: maac.trainer.DEFAULT_CONFIG,
    postprocess_fn=postprocess_trajectory,
    # view_requirement_fn=view_requirement_fn,
    mixins=[TargetNetworkMixin, ],
    after_init=setup_late_mixins,
    extra_grad_process_fn=apply_grad_clipping,
    stats_fn=stats,
    loss_fn=loss_fn,
)
