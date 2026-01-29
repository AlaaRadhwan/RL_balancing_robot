import time
import numpy as np
import gymnasium as gym

from envs.a1_env import A1Env

env = A1Env(render_mode="human")

obs, info = env.reset()
env.render()

zero_action = np.zeros(env.action_space.shape)

for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(zero_action)

    if terminated or truncated:
        print("Episode ended")
        break

    # time.sleep(env.control_dt)

env.close()
