import math
from itertools import chain
from typing import Dict, List

import gym
from gym.spaces import Box, MultiDiscrete
from ray.rllib import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch, TensorType
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.typing import ModelConfigDict

torch, nn = try_import_torch()
F = nn.functional


class MAACTorchModel(TorchModelV2, nn.Module):
    """Extension of standard TorchModelV2 to MAAC."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str = 'MAACTorchModel'):
        nn.Module.__init__(self)
        super(MAACTorchModel, self).__init__(obs_space, action_space,
                                             num_outputs,
                                             model_config, name)

        assert hasattr(self.obs_space, "original_space") and isinstance(
                self.obs_space.original_space, gym.spaces.Dict)

        original_space = self.obs_space.original_space
        self.true_obs_space = original_space['obs']

        if not isinstance(self.true_obs_space, Box):
            raise UnsupportedSpaceException(
                "Space {} is not supported as observation.".format(
                    self.true_obs_space)
            )

        if not isinstance(action_space, MultiDiscrete):
            raise UnsupportedSpaceException(
                "Space {} is not supported as action.".format(self.action_space)
            )

        assert len(
            self.true_obs_space.shape) == 2, "Observation space is supposed to " \
                                             "have 2 dimensions."

        self.nbr_agents, input_dim = self.true_obs_space.shape
        self.nbr_actions = int(self.action_space.nvec[0])

        self.num_heads = model_config["nbr_heads"]
        self.att_dim = model_config["critic_emb_dim"] // self.num_heads

        self.actor = self.create_actor()
        self.target_actor = self.create_actor()
        self.critic = self.create_critic()

    def create_critic(self):
        # state encoder
        obs_dim = self.true_obs_space.shape[1]
        emb_dim = self.model_config.get("critic_emb_dim")
        obs_encoder = []
        obs_act_encoder = []
        q_decoder = []
        for i in range(self.nbr_agents):
            obs_encoder_layers = []
            obs_act_encoder_layers = []
            if self.model_config.get("batch_norm"):
                obs_encoder_layers.append(
                    nn.BatchNorm1d(obs_dim, affine=False)
                )
                obs_act_encoder_layers.append(
                    nn.BatchNorm1d(obs_dim + self.nbr_actions, affine=False)
                )
            obs_encoder_layers.append(
                SlimFC(in_size=obs_dim,
                       out_size=emb_dim,
                       initializer=normc_initializer(1.0),
                       activation_fn=nn.LeakyReLU,
                       )
            )
            obs_act_encoder_layers.append(
                SlimFC(in_size=obs_dim + self.nbr_actions,
                       out_size=emb_dim,
                       initializer=normc_initializer(1.0),
                       activation_fn=nn.LeakyReLU,
                       )
            )
            obs_encoder.append(nn.Sequential(*obs_encoder_layers))
            obs_act_encoder.append(nn.Sequential(*obs_act_encoder_layers))
            q_decoder.append(nn.Sequential(
                SlimFC(in_size=2 * emb_dim,
                       out_size=emb_dim,
                       initializer=normc_initializer(1.),
                       activation_fn=nn.LeakyReLU),
                SlimFC(in_size=emb_dim,
                       out_size=self.nbr_actions,
                       initializer=normc_initializer(1.))
            ))

        obs_encoder = nn.ModuleList(obs_encoder)
        obs_act_encoder = nn.ModuleList(obs_act_encoder)
        q_decoder = nn.ModuleList(q_decoder)

        key_extractor = SlimFC(in_size=emb_dim,
                               out_size=emb_dim,
                               use_bias=False,
                               initializer=normc_initializer(1.), )
        query_extractor = SlimFC(in_size=emb_dim,
                                 out_size=emb_dim,
                                 use_bias=False,
                                 initializer=normc_initializer(1.), )
        value_extractor = SlimFC(in_size=emb_dim,
                                 out_size=emb_dim,
                                 use_bias=True,
                                 # In the original code it is set to true in the paper it is false
                                 initializer=normc_initializer(1.),
                                 activation_fn=nn.LeakyReLU, )

        critic = nn.ModuleDict({
            "obs_encoder": obs_encoder,
            "obs_act_encoder": obs_act_encoder,
            "key_extractor": key_extractor,
            "query_extractor": query_extractor,
            "value_extractor": value_extractor,
            "q_decoder": q_decoder})
        return critic

    def get_attention_parameters(self):
        return chain(
            self.critic["key_extractor"].parameters(),
            self.critic["query_extractor"].parameters(),
            self.critic["value_extractor"].parameters(),
            self.critic["obs_act_encoder"].parameters(),
        )

    def create_actor(self):
        nbr_agents = self.nbr_agents
        input_size = self.true_obs_space.shape[1]

        models = []
        activation = nn.LeakyReLU
        hidden_units = self.model_config.get("actor_hiddens", )
        hidden_units = list(hidden_units) + [self.nbr_actions]
        for i in range(nbr_agents):
            prev_layer_size = input_size
            layers = []
            if self.model_config.get("batch_norm"):
                layers.append(nn.BatchNorm1d(prev_layer_size, affine=False))
            for size in hidden_units:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=size,
                        initializer=normc_initializer(1.0),
                        activation_fn=activation
                    )
                )
                prev_layer_size = size

            models.append(nn.Sequential(*layers))

        return nn.ModuleList(models)

    def get_q_values(self, input_dict: Dict[str, TensorType]) -> TensorType:
        obs_encoder = self.critic["obs_encoder"]
        obs_act_encoder = self.critic["obs_act_encoder"]
        key_extractor = self.critic["key_extractor"]
        query_extractor = self.critic["query_extractor"]
        value_extractor = self.critic["value_extractor"]
        q_decoder = self.critic["q_decoder"]

        nbr_agents = self.nbr_agents
        emb_dim = self.model_config.get("critic_emb_dim")
        input_dict[SampleBatch.OBS] = restore_original_dimensions(
            input_dict[SampleBatch.OBS], self.obs_space, self.framework)
        obs = input_dict[SampleBatch.OBS]['obs']
        act = input_dict[SampleBatch.ACTIONS]
        act_one_hot = F.one_hot(act, self.nbr_actions).to(obs.dtype)
        obs_act = torch.cat([obs, act_one_hot], dim=-1)
        B = obs.shape[0]

        obs_emb = torch.stack([
            obs_encoder_(obs_.squeeze(1)) for obs_encoder_, obs_ in
            zip(obs_encoder, obs.chunk(nbr_agents, 1))
        ], dim=1)
        obs_act_emb = torch.stack([
            obs_act_encoder_(obs_act_.squeeze(1)) for obs_act_encoder_, obs_act_
            in zip(obs_act_encoder, obs_act.chunk(nbr_agents, 1))
        ], dim=1)

        key = key_extractor(obs_act_emb)
        value = value_extractor(obs_act_emb)
        query = query_extractor(obs_emb)
        query = query.view(B, nbr_agents, 1, self.num_heads, self.att_dim)
        key = key.view(B, 1, nbr_agents, self.num_heads, self.att_dim)
        value = value.view(B, 1, nbr_agents, self.num_heads, self.att_dim)
        att_weights_logits = (query * key).sum(-1)
        # mask self attention
        mask = torch.eye(self.nbr_agents, dtype=key.dtype,
                         device=key.device).view(1, self.nbr_agents,
                                                 self.nbr_agents, 1)
        mask_inf = torch.clamp(torch.log(1 - mask), FLOAT_MIN, FLOAT_MAX)
        att_weights_logits_masked = att_weights_logits + mask_inf
        scaled_att_weights_logits_masked = att_weights_logits_masked / math.sqrt(
            self.att_dim)
        log_att_weights = F.log_softmax(scaled_att_weights_logits_masked, dim=2)

        att_weights = torch.exp(log_att_weights)

        attention_entropy = - (log_att_weights * att_weights).sum(2)

        attended_values = (att_weights.unsqueeze(-1) * value).sum(2).view(B,
                                                                          nbr_agents,
                                                                          emb_dim)
        encoded = torch.cat([obs_emb, attended_values], dim=-1)

        q_values = torch.stack([
            q_decoder_(encoded_.squeeze(1)) for q_decoder_, encoded_ in
            zip(q_decoder, encoded.chunk(nbr_agents, 1))
        ], dim=1)
        reg = 1e-3 * (torch.pow(att_weights_logits, 2.) * (1 - mask)).sum(
            (2, 3)) / (nbr_agents - 1)

        return q_values, reg, attention_entropy

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType], seq_lens: TensorType) -> (
            TensorType, List[TensorType]):

        obs = input_dict[SampleBatch.OBS]['obs']
        B = obs.shape[0]
        obs = torch.split(obs, 1, dim=-2)
        logits = [actor_(o.squeeze(-2)) for actor_, o in zip(self.actor, obs)]

        return logits.view(B, self.num_outputs), []

    def actor_variables(self, as_dict: bool = False):
        if as_dict:
            return self.actor.state_dict()
        return self.actor.parameters()

    def actor_variable_per_agent(self, agent: int, as_dict=False):
        assert 0 <= agent < self.nbr_agents
        if as_dict:
            return self.actor[agent].state_dict()
        return self.actor[agent].parameters()

    def critic_variables(self, as_dict: bool = False):
        if as_dict:
            return self.critic.state_dict()
        return self.critic.parameters()
