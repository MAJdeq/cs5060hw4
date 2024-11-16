import gymnasium
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

mean, std = evaluate_policy(model, model.get_env())
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
