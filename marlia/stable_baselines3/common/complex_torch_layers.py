from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class MultiNatureCNN(BaseFeaturesExtractor):
    """
    Combined feature extractor for (layers, width, height, channels) observation spaces.
    Input from each layer is fed through a separate NatureCNN submodule, 
    and the output features are concatenated and fed through additional MLP network ("combined").
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(self, observation_space: gym.spaces.Box, cnn_output_dim: int = 256):
        # Output feature dimensionality is the sum of the output dimensions of each CNN submodule
        features_dim = observation_space.shape[0] * cnn_output_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Create a CNN submodule for each layer
        layer_observation_space = gym.spaces.Box(
            low=observation_space.low[0][0][0][0],
            high=observation_space.high[0][0][0][0],
            shape=observation_space.shape[1:],
            dtype=observation_space.dtype,
        )
        extractors = []
        for _ in range(observation_space.shape[0]):
            extractors.append(NatureCNN(layer_observation_space, features_dim=cnn_output_dim))
        self.extractors = nn.ModuleList(extractors)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Split the observations into a list, one for each layer
        observations = th.split(observations, 1, dim=1)
        # Apply each CNN submodule to its corresponding layer
        encoded_tensor_list = []
        for layer, extractor in enumerate(self.extractors):
            # Squeeze the layer dimension
            obs_layer = th.squeeze(observations[layer], dim=1)
            encoded_tensor_list.append(extractor(obs_layer))
        return th.cat(encoded_tensor_list, dim=1)
