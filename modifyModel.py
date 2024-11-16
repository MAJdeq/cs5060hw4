from typing import Tuple, Callable

import gymnasium
from gymnasium import RewardWrapper
from gymnasium.vector.utils import spaces
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from stable_baselines3 import PPO
import time

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 100), nn.ReLU(),
            nn.Linear(100, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 100), nn.ReLU(),
            nn.Linear(100, last_layer_dim_vf)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

class CustomCartPoleReward(RewardWrapper):
    def __init__(self, env):
        super(CustomCartPoleReward, self).__init__(env)

    def reward(self, reward):
        x, x_dot, theta, theta_dot = self.env.state
        reward -= np.abs(theta)  # Penalize for angle from upright
        if np.abs(x) > 2:
            reward -= 10
        return reward

# Initialize the custom environment
env = CustomCartPoleReward(gymnasium.make("CartPole-v1",render_mode="rgb_array"))

model = PPO(CustomActorCriticPolicy, env, policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[]), verbose=1)

start = time.time()
model.learn(total_timesteps=25000)
end = time.time()

mean, std = evaluate_policy(model, model.get_env())
print(f"Total train time: {end-start}s")
print(f"{mean=}, {std=}")

num_eval_episodes = 4
env = RecordVideo(env, video_folder="cartpole-agent", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env)

for episode_num in range(num_eval_episodes):
    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated
env.close()

