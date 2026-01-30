import gymnasium as gym
import time

env = gym.make(
    "Humanoid-v5",
    render_mode="human"
)

obs, info = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()
