import argparse
import os
import numpy as np
import time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

from envs.robust_env import VelocityEnv

from evdev import InputDevice, list_devices, ecodes
import select

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

parser.add_argument(
    "--control",
    choices=["random", "ps5"],
    default="random",
    help="Command source: random or PS5 controller"
)

args = parser.parse_args()


# --------------------------------------------------
# Paths
# --------------------------------------------------
RUN_NAME = "balance_env2.1_yaw"
STEP = args.step
ckpt_name_prefix = "ppo_balance"
MODEL_PATH = f"./checkpoints/{RUN_NAME}/{ckpt_name_prefix}_{STEP}_steps.zip"
VEC_PATH = f"./checkpoints/{RUN_NAME}/vec_normalize_{STEP}_steps.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(VEC_PATH):
    raise FileNotFoundError(f"VecNormalize not found: {VEC_PATH}")


# --------------------------------------------------
# Environment factory
# --------------------------------------------------
def make_env():
    env = VelocityEnv(render_mode="human")
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


ps5 = None
if args.control == "ps5":
    devices = [InputDevice(path) for path in list_devices()]
    for dev in devices:
        if "Wireless Controller" in dev.name or "DualSense" in dev.name:
            ps5 = dev
            ps5.grab()
            print(f"[INFO] Using controller: {dev.name}")
            break

    if ps5 is None:
        raise RuntimeError("PS5 controller not found")

# PS5 axis state
lx = 0.0   # left stick horizontal
ly = 0.0   # left stick vertical

MAX_V_CMD = 1.0      # m/s (your scale)
MAX_YAW_CMD = 1.0    # rad/s equivalent


def read_ps5():
    global lx, ly

    if ps5 is None:
        return

    r, _, _ = select.select([ps5.fd], [], [], 0)
    if not r:
        return

    for event in ps5.read():
        if event.type != ecodes.EV_ABS:
            continue

        # LEFT STICK X  → yaw
        if event.code == ecodes.ABS_RX:
            lx = (event.value - 128) / 128.0

        # LEFT STICK Y  → velocity
        elif event.code == ecodes.ABS_Y:
            ly = (event.value - 128) / 128.0
# --------------------------------------------------
# Push configuration
# --------------------------------------------------
ENABLE_PUSH = args.push

PUSH_INTERVAL = 300        # control steps
PUSH_DURATION = 25         # control steps
BASE_FORCE = 5.0           # Newtons

CMD_INTERVAL = 300

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

        if args.control == "ps5":
            read_ps5()

            # Forward/backward on left stick Y
            real_env.v_cmd = -ly * MAX_V_CMD

            # Turn on left stick X
            real_env.yaw_cmd = lx * MAX_YAW_CMD

        else:
            if t % CMD_INTERVAL == 0:
                real_env.v_cmd = np.random.uniform(-1.0, 1.0)
                real_env.yaw_cmd = np.random.uniform(-1.0, 1.0)

        if abs(real_env.theta_est) > real_env.THETA_SAFE:
            real_env.yaw_cmd = real_env.yaw_est

        if abs(real_env.v_cmd) < 0.1:
            real_env.v_cmd = 0.0

        if abs(real_env.yaw_cmd) < 0.1:
            real_env.yaw_cmd = 0.0
        
        # Optional light diagnostics
        if t % 100 == 0:
            base_quat = real_env.data.qpos[3:7]
            print(
                f"[t={t:4d}] reward={reward[0]:+.3f}"
            )

        print(
            f"[CMd] v_cmd: {real_env.v_cmd:+.3f},   v_f: {real_env.forward_velocity():+.3f}"
            f"[CMD] y_cmd: {real_env.yaw_cmd:+.3f} y_e: {real_env.yaw_est:+.3f}"
        )

        t += 1

except KeyboardInterrupt:
    print("\n[INFO] Evaluation stopped by user")

finally:
    env.close()
