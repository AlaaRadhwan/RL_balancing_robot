import argparse
import os
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

from envs.a1_env import A1Env


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Interactive visual evaluation for A1 standing policy"
)


parser.add_argument(
    "--step",
    type=int,
    required=True,
    help="Checkpoint timestep to load (e.g. 500000)"
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=1500,
    help="Max steps per episode before reset"
)

parser.add_argument(
    "--push",
    action="store_true",
    help="Enable external pushes"
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------
RUN_NAME = "a1_env3.5_walking_v1.1"
STEP = args.step

MODEL_PATH = f"./checkpoints/{RUN_NAME}/a1_standing_v0_{STEP}_steps.zip"
VEC_PATH = f"./checkpoints/{RUN_NAME}/vec_normalize_{STEP}_steps.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(VEC_PATH):
    raise FileNotFoundError(f"VecNormalize not found: {VEC_PATH}")


# --------------------------------------------------
# Environment factory
# --------------------------------------------------
def make_env():
    env = A1Env(render_mode="human")
    env = TimeLimit(env, max_episode_steps=args.max_steps)
    return env


# --------------------------------------------------
# Create environment
# --------------------------------------------------
env = DummyVecEnv([make_env])

# Load normalization ("the glasses")
env = VecNormalize.load(VEC_PATH, env)
env.training = False
env.norm_reward = False

# Load policy
model = PPO.load(MODEL_PATH, device="cpu")

obs = env.reset()

real_env = env.envs[0].unwrapped


# --------------------------------------------------
# Push configuration
# --------------------------------------------------
ENABLE_PUSH = args.push

PUSH_INTERVAL = 300        # control steps
PUSH_DURATION = 25         # control steps
BASE_FORCE = 5.0           # Newtons

push_steps_left = 0
push_force = np.zeros(2)


def maybe_apply_push(t):
    global push_steps_left, push_force

    if not ENABLE_PUSH:
        return

    if push_steps_left == 0 and t % PUSH_INTERVAL == 0:
        angle = np.random.uniform(0, 2 * np.pi)
        push_force = BASE_FORCE * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        push_steps_left = PUSH_DURATION
        print(f"[PUSH] force = {push_force}")

    if push_steps_left > 0:
        real_env.data.xfrc_applied[
            real_env.trunk_body_id, :2
        ] += push_force
        push_steps_left -= 1


# --------------------------------------------------
# Main evaluation loop
# --------------------------------------------------
print("\n[INFO] Starting A1 standing visual evaluation")
print("      Press Ctrl+C to exit\n")

t = 0

try:
    while True:
        maybe_apply_push(t)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            print("[RESET] Episode ended")
            obs = env.reset()
            t = 0
            push_steps_left = 0
            continue

        # Optional light diagnostics
        if t % 100 == 0:
            base_quat = real_env.data.qpos[3:7]
            print(
                f"[t={t:4d}] reward={reward[0]:+.3f}"
            )

        t += 1

except KeyboardInterrupt:
    print("\n[INFO] Evaluation stopped by user")

finally:
    env.close()
