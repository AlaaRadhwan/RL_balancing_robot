import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
import mujoco

# Import the correct V2 environment
from envs.velocity_env_v3_2_forces import VelocityEnv

# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------
RUN_NAME = "Balance_v3.6_fcs_w"
parser = argparse.ArgumentParser(description="Evaluate velocity-tracking policy")


parser.add_argument(
    "--ckpt",
    type=str,
    default=f"./checkpoints/{RUN_NAME}/best/best_model.zip",
    help="Path to model checkpoint (.zip)"
)

parser.add_argument(
    "--stats",
    type=str,
    default=f"./checkpoints/{RUN_NAME}/vec_normalize.pkl",
    help="Path to normalization stats (.pkl)"
)

parser.add_argument(
    "--no-forces",
    action="store_true",
    help="Disable external disturbance forces during evaluation"
)

# NEW: Control episode length
parser.add_argument(
    "--max-steps",
    type=int,
    default=1000, # 1000 steps = 20 seconds at 50Hz
    help="Maximum steps per episode before forced reset"
)

args = parser.parse_args()

if args.ckpt:
    TIMESTEP = int(args.ckpt)
else:
    TIMESTEP = 1_000_000

MODEL_PATH = (
    f"./checkpoints/{RUN_NAME}/ppo_velocity_{TIMESTEP}_steps.zip"
)

VEC_PATH = (
    f"./checkpoints/{RUN_NAME}/vec_normalize_{TIMESTEP}_steps.pkl"
)
# --------------------------------------------------
# 1. Setup Environment
# --------------------------------------------------
# We wrap the env in TimeLimit so it returns 'done=True' (truncated) automatically
def make_env():
    env = VelocityEnv(render_mode="human", enable_disturbance=True)
    return TimeLimit(env, max_episode_steps=args.max_steps)

env = DummyVecEnv([make_env])

# --------------------------------------------------
# 2. Load Normalization Stats ("The Glasses")
# --------------------------------------------------
if os.path.exists(VEC_PATH):
    print(f"[INFO] Loading normalization stats from {args.stats}")
    # env = VecNormalize.load(args.stats, env)
    env = VecNormalize.load(VEC_PATH, env)
    print(f"[INFO] model path: {VEC_PATH}")

    # CRITICAL: Freeze stats so evaluation doesn't mess up the running average
    env.training = False 
    env.norm_reward = False 
else:
    print(f"[WARNING] No stats found at {args.stats}!")
    print("[WARNING] The robot will likely fall immediately.")

# --------------------------------------------------
# 3. Load Model
# --------------------------------------------------
# print(f"[INFO] Loading model: {args.ckpt}")
print(f"[INFO] Loading model: {MODEL_PATH}")

# model = PPO.load(args.ckpt, device="cpu")
model = PPO.load(MODEL_PATH, device="cpu")

obs = env.reset()

# --------------------------------------------------
# Cache joint qpos addresses (ONCE)
# --------------------------------------------------
real_env = env.envs[0].unwrapped

# joint_names = [
#     "left_hip_joint",
#     "right_hip_joint",
#     "left_knee_joint",
#     "right_knee_joint",
# ]

# joint_qpos_adrs = {}
# for name in joint_names:
#     jid = mujoco.mj_name2id(real_env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
#     joint_qpos_adrs[name] = real_env.model.jnt_qposadr[jid]

# # Print initial joint configuration
# print("\n[RESET STATE] Joint angles at spawn:")
# for name, adr in joint_qpos_adrs.items():
#     print(f"  {name}: {real_env.data.qpos[adr]:+.3f} rad")


# Optional: Force render once at start to see the spawn state
# env.render()
env.envs[0].render()

# --------------------------------------------------
# 4. Evaluation Loop
# --------------------------------------------------
# Disturbance Config

# BASE_FORCE = 7.0
FORCE_JITTER = 0.2
DIST_STEPS = 40
PUSH_INTERVAL = 350
disturb_steps_left = 0
next_push_t = 200
disturb_force = np.zeros(2)

enable_forces = not args.no_forces
v_err_hist = []

print(f"\n[INFO] Starting Simulation (Max Steps: {args.max_steps})... Press Ctrl+C to stop.")

try:
    t = 0

    PUSH_INTERVAL = 250        # control steps
    DIST_DURATION = 20         # control steps
    BASE_FORCE = 5.0

    # --- Velocity command schedule ---
    CMD_INTERVAL = 500      # control steps (~6s at 50Hz)
    CMD_MIN = -0.5
    CMD_MAX = 1.5
    
    while True:
        # Access the real underlying MujocoEnv
        real_env = env.envs[0].unwrapped # .unwrapped gets past TimeLimit to the real env

        # real_env.v_cmd = 0.7
        # real_env.yaw_cmd = 1.0

        # --- Disturbance Logic ---

        if enable_forces:
            if t % PUSH_INTERVAL == 0:
                angle = np.random.uniform(0, 2*np.pi)
                force_xy = BASE_FORCE * np.array([np.cos(angle), np.sin(angle)])
                real_env.trigger_disturbance(force_xy, DIST_DURATION)

                print(F"[PUSH] t={t} | force={force_xy} N")

        # --- Velocity Command Update ---
        if t % CMD_INTERVAL == 0:
            v_cmd = np.random.uniform(CMD_MIN, CMD_MAX)
            max_yaw = max(0.3, abs(v_cmd))
            yaw_cmd = np.random.uniform(-max_yaw, max_yaw)


            real_env.v_cmd = v_cmd
            real_env.yaw_cmd = yaw_cmd

            print(
                f"[CMD] t={t} | v_cmd={v_cmd:+.2f} m/s | yaw_cmd={yaw_cmd:+.2f} rad/s"
            )

        # --- Predict & Step ---
        action, _ = model.predict(obs, deterministic=True)
        # action = np.zeros((1, 10)) 
        obs, reward, done, infos = env.step(action)

        # --- Logging ---
        # done[0] is True if robot falls OR max_steps reached
        if done[0]:
            print(f"[RESET] Episode finished. Last v_cmd: {real_env.v_cmd:.2f}")
            obs = env.reset()
            v_err_hist.clear()
            t = 0
            next_push_t = 200
            disturb_steps_left = 0 # Clear any active push

        yaw_rate = real_env.data.sensordata[real_env.gyro_adr + 2]

        print(
            f"[INFO] v: {real_env.forward_velocity():+.3f} / {real_env.v_cmd:+.3f} m/s | "
            f"w: {yaw_rate:+.3f} / {real_env.yaw_cmd:+.3f} rad/s"
        )

        v_actual = real_env.forward_velocity()
        v_cmd = real_env.v_cmd
        v_error = v_actual - v_cmd

        v_err_hist.append(abs(v_error))
        if len(v_err_hist) > 500: v_err_hist.pop(0)


        # --- Joint Angle Monitoring ---
        # if t % 50 == 0:  # every 1 second at 50 Hz
        #     print(f"\n[t = {t}] Joint angles:")
        #     for name, adr in joint_qpos_adrs.items():
        #         print(f"  {name}: {real_env.data.qpos[adr]:+.3f} rad")

        # Sleep to match Real-Time
        time.sleep(real_env.control_dt)
        t += 1

except KeyboardInterrupt:
    print("\n[INFO] Evaluation stopped by user.")
finally:
    env.close()