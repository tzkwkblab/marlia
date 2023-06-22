from typing import Any, Dict, Optional, Type, Union, List, Tuple

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.torch_layers import (
    create_mlp,
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from ..complex_torch_layers import MultiNatureCNN
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

class PredictiveRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = True,
        enable_critic_lstm: bool = False,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        auxnet_arch: List[int] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )

        if isinstance(self.action_space, spaces.Discrete):
            self.action_flat_dim = self.action_space.n
        else:
            self.action_flat_dim = np.prod(action_space.shape)
        aux_input_dim = lstm_hidden_size + self.action_flat_dim
        aux_output_dim = np.prod(observation_space.shape)
        if auxnet_arch == None:
            auxnet_arch = [int((aux_output_dim + aux_input_dim) / 2)]
        self.auxnet = create_mlp(aux_input_dim, aux_output_dim, auxnet_arch)
        self.auxnet = th.nn.Sequential(*self.auxnet)

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        ######################################
        # Predict the next observation
        if isinstance(self.action_space, spaces.Discrete):
            aux_input_action = th.nn.functional.one_hot(actions, num_classes=self.action_flat_dim)
        else:
            aux_input_action = actions
        aux_input = th.cat((latent_pi, aux_input_action), axis=1)
        obs_predict = self.auxnet(aux_input)
        obs_predict = obs_predict.reshape((-1,) + self.observation_space.shape)
        ######################################

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), obs_predict
    

class PredictiveRecurrentCnnActorCriticPolicy(PredictiveRecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = MultiNatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = True,
        enable_critic_lstm: bool = False,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
        auxnet_arch: List[int] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            shared_lstm,
            enable_critic_lstm,
            lstm_kwargs,
        )