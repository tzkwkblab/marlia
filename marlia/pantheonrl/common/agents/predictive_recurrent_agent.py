from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

from collections import deque
import time

import numpy as np
import torch as th
from copy import deepcopy

import gym
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Space

from pantheonrl.common.util import action_from_policy, clip_actions, resample_noise
from pantheonrl.common.trajsaver import TransitionsMinimal
from pantheonrl.common.agents import OnPolicyAgent

from stable_baselines3.common import policies
from stable_baselines3.common.utils import (
    configure_logger,
    should_collect_more_steps
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import safe_mean, obs_as_tensor

from sb3_contrib.common.recurrent.type_aliases import RNNStates


class PredictiveRecurrentOnPolicyAgent(OnPolicyAgent):
    """
    Agent representing an on-policy learning algorithm (ex: A2C/PPO).

    The `get_action` and `update` functions are based on the `learn` function
    from ``OnPolicyAlgorithm``.

    :param model: Model representing the agent's learning algorithm
    """

    def __init__(self,
                 model: OnPolicyAlgorithm,
                 log_interval=None,
                 tensorboard_log=None,
                 tb_log_name="OnPolicyAgent"):
        self.model = model
        self._last_episode_starts = [True]
        self.n_steps = 0
        self.values: th.Tensor = th.empty(0)

        self.model.set_logger(configure_logger(
            self.model.verbose, tensorboard_log, tb_log_name))

        self.name = tb_log_name
        self.num_timesteps = 0
        self.log_interval = log_interval or (1 if model.verbose else None)
        self.iteration = 0
        self.model.ep_info_buffer = deque([{"r": 0, "l": 0}], maxlen=100)
        
        lstm = self.model.policy.lstm_actor
        single_hidden_state_shape = (lstm.num_layers, self.model.n_envs, lstm.hidden_size)
        self._last_lstm_states = RNNStates(
            (
                th.zeros(single_hidden_state_shape, device=self.model.policy.device),
                th.zeros(single_hidden_state_shape, device=self.model.policy.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.model.policy.device),
                th.zeros(single_hidden_state_shape, device=self.model.policy.device),
            ),
        )

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        When `record` is True, the agent saves the last transition into its
        buffer. It also updates the model if the buffer is full.

        :param obs: The observation to use
        :param record: Whether to record the obs, action (True when training)
        :returns: The action to take
        """
        buf = self.model.rollout_buffer
        
        buf.next_observations[buf.pos-1] = obs

        # train the model if the buffer is full
        if record and self.n_steps >= self.model.n_steps:
            buf.compute_returns_and_advantage(
                last_values=self.values,
                dones=self._last_episode_starts[0]
            )

            if self.log_interval is not None and \
                    self.iteration % self.log_interval == 0:
                self.model.logger.record(
                    "name", self.name, exclude="tensorboard")
                self.model.logger.record(
                    "time/iterations", self.iteration, exclude="tensorboard")

                if len(self.model.ep_info_buffer) > 0 and \
                        len(self.model.ep_info_buffer[0]) > 0:
                    last_exclude = self.model.ep_info_buffer.pop()
                    rews = [ep["r"] for ep in self.model.ep_info_buffer]
                    lens = [ep["l"] for ep in self.model.ep_info_buffer]
                    self.model.logger.record(
                        "rollout/ep_rew_mean", safe_mean(rews))
                    self.model.logger.record(
                        "rollout/ep_len_mean", safe_mean(lens))
                    self.model.ep_info_buffer.append(last_exclude)

                self.model.logger.record(
                    "time/total_timesteps", self.num_timesteps,
                    exclude="tensorboard")
                self.model.logger.dump(step=self.num_timesteps)

            self.model.train()
            self.iteration += 1
            buf.reset()
            self.n_steps = 0

        resample_noise(self.model, self.n_steps)

        actions, values, log_probs, self._last_lstm_states = self.action_from_recurrent_policy(obs, self.model.policy, self._last_lstm_states, self._last_episode_starts)

        # modify the rollout buffer with newest info
        if record:
            lastinfo = self.model.ep_info_buffer.pop()
            lastinfo["l"] += 1
            self.model.ep_info_buffer.append(lastinfo)

            obs_shape = self.model.policy.observation_space.shape
            act_shape = self.model.policy.action_space.shape
            buf.add(
                np.reshape(obs, (1,) + obs_shape),
                np.reshape(actions, (1,) + act_shape),
                [0],
                self._last_episode_starts,
                values,
                log_probs,
                next_obs = buf.next_observations[buf.pos-1],
                lstm_states = self._last_lstm_states
            )

        self.n_steps += 1
        self.num_timesteps += 1
        self.values = values
        return clip_actions(actions, self.model)[0]

    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information.

        The rewards are added to buffer entry corresponding to the most recent
        recorded action.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
        buf = self.model.rollout_buffer
        self._last_episode_starts = [done]
        buf.rewards[buf.pos - 1][0] += reward
        lastinfo = self.model.ep_info_buffer.pop()
        lastinfo["r"] += reward
        self.model.ep_info_buffer.append(lastinfo)
        if done:
            self.model.ep_info_buffer.append({"r": 0, "l": 0})

    def learn(self, **kwargs) -> None:
        """ Call the model's learn function with the given parameters """
        self.model._custom_logger = False
        self.model.learn(**kwargs)
    
    def action_from_recurrent_policy(self, obs, policy, _last_lstm_states, _last_episode_starts):
        """
        Return the action, values, and log_probs given an observation and policy

        : param obs: Numpy array representing the observation
        : param policy: The actor-critic policy

        : returns: The action, values, and log_probs from the policy
        """
    
        obs = obs.reshape((-1,) + policy.observation_space.shape)
    
        lstm_states = deepcopy(_last_lstm_states)
    
        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs, policy.device)
            episode_starts = th.tensor(_last_episode_starts, dtype=th.float32, device=policy.device)
            actions, values, log_probs, lstm_states = policy.forward(obs_tensor, lstm_states, episode_starts)

        return actions.cpu().numpy(), values, log_probs, lstm_states
