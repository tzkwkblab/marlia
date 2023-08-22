import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import obs_as_tensor

# Memoize decorator for dynamic inheritance of BaseAlgorithm classes
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        return self.memo.setdefault(args, self.f(*args))

# Random Network Distillation
# Wrapper for BaseAlgorithm classes that adds a curiosity module based on the RND method
@Memoize
def RND(base: BaseAlgorithm):
    # Dynamic inheritance
    class RNDWrapper(base):
        def _setup_model(self, *args, **kwargs):
            super()._setup_model(*args, **kwargs)
            # Initialize random target network where the weights are frozen
            # The input is the flattened observation and the output size is self.rep_size
            flattened_obs_dim = np.prod(self.observation_space.shape)
            middle_rep_size = int(flattened_obs_dim / 2)
            self.rep_size = int(flattened_obs_dim / 4)
            self.target_network = th.nn.Sequential(
                th.nn.Linear(flattened_obs_dim, middle_rep_size),
                th.nn.ReLU(),
                th.nn.Linear(middle_rep_size, self.rep_size),
            ).to(self.device)
            # Initialize predictor network
            self.predictor_network = th.nn.Sequential(
                th.nn.Linear(flattened_obs_dim, middle_rep_size),
                th.nn.ReLU(),
                th.nn.Linear(middle_rep_size, self.rep_size),
            ).to(self.device)
            # Freeze the target network
            for param in self.target_network.parameters():
                param.requires_grad = False
            # Initialize the optimizer for the predictor network
            self.predictor_optimizer = th.optim.Adam(self.predictor_network.parameters(), lr=1e-4)

        def get_intrinsic_reward(self, obs):
            with th.no_grad():
                obs_tensor = obs_as_tensor(obs.flatten(), self.device)
                # Calculate the output of the target network and the predictor network
                target_output = self.target_network(obs_tensor)
                predictor_output = self.predictor_network(obs_tensor)
                # Calculate the L2 loss between the target output and the predictor output
                curiosity_reward = th.nn.functional.mse_loss(
                    predictor_output, target_output.detach(), reduction='sum'
                )
                # Return the intrinsic reward
                reward = curiosity_reward.detach().cpu().numpy()
                return reward
        
        def train(self):
            # Call the train method of the base class
            super().train()
            # Update the target network
            n_epochs = 1
            curiosity_losses = []
            for epoch in range(n_epochs):
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    target_output = self.target_network(rollout_data.observations.flatten(1))
                    predictor_output = self.predictor_network(rollout_data.observations.flatten(1))
                    # Calculate the L2 loss between the target output and the predictor output
                    curiosity_loss_sum = th.nn.functional.mse_loss(
                        predictor_output, target_output.detach(), reduction='sum'
                    )
                    curiosity_loss = curiosity_loss_sum / rollout_data.observations.shape[0]
                    curiosity_losses.append(curiosity_loss.item())
                    # Optimizer step for the predictor network
                    self.predictor_optimizer.zero_grad()
                    curiosity_loss.backward()
                    self.predictor_optimizer.step()
            self.logger.record("train/curiosity_loss", np.mean(curiosity_losses))

        
    return RNDWrapper
