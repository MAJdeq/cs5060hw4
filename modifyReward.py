import time

import gymnasium
from gymnasium import RewardWrapper
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

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
env = CustomCartPoleReward(gymnasium.make("CartPole-v1", render_mode="rgb_array"))

model = PPO("MlpPolicy", env, verbose=1)
start = time.time()
model.learn(total_timesteps=50000)
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

