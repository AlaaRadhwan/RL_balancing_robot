import os
import multiprocessing
import numpy as np
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Import your corrected environment
from envs.velocity_env_v3_2_forces import VelocityEnv

# --------------------------------------------------
# 0. Configuration & Hyperparameters
# --------------------------------------------------
RUN_NAME = "Balance_v3.6_fcs_w"
PREV_RUN_NAME = "Balance_v3.5_fcs_w"
NUM_ENVS = 8  # Ryzen 7 5000 has 8+ cores. 8 is a safe sweet spot.
SEED = 42

# Base directories
LOG_DIR = f"./logs/{RUN_NAME}"
# LOG_DIR_RESUM = f"./logs/{RUN_NAME}_resume"
CHECKPOINT_DIR = f"./checkpoints/{RUN_NAME}"
PREV_CKPT_DIR = f"./checkpoints/{PREV_RUN_NAME}"

# Optimized Hyperparameters for Velocity Control (Standard RL Zoo style)
HYPERPARAMS = {
    "n_steps": 2048,           # Steps per environment before update
    "batch_size": 2048,        # Larger batch size for stable gradients
    "learning_rate": 3e-4,     # Standard stable LR
    "gamma": 0.99,             # Look ahead ~100 steps (2 seconds)
    "gae_lambda": 0.95,        # Generalized Advantage Estimation
    "clip_range": 0.2,         # PPO clipping
    "ent_coef": 0.0,           # Let the noise come from std_init, not forced entropy
    "policy_kwargs": dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]), # Stronger brain
        log_std_init=-1.5,       # Start with small exploration noise (prevent instant death)
        ortho_init=True,       # Orthogonal initialization (standard for PPO)
    ),
}

RESUME = True

RESUME_STEP = 11_000_000

RESUME_MODEL_PATH = (
    f"{PREV_CKPT_DIR}/ppo_velocity_{RESUME_STEP}_steps.zip"
)

RESUME_VEC_PATH = (
    f"{PREV_CKPT_DIR}/vec_normalize_{RESUME_STEP}_steps.pkl"
)

# --------------------------------------------------
# 1. Custom Callback to Save "Glasses" (VecNormalize)
# --------------------------------------------------
class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # num_timesteps is GLOBAL timesteps (correct)
            path = os.path.join(
                self.save_path,
                f"vec_normalize_{self.num_timesteps}_steps.pkl"
            )
            self.training_env.save(path)

            if self.verbose > 0:
                print(f"[VecNormalize] Saved stats to {path}")

        return True

# --------------------------------------------------
# 2. Environment Setup Helper
# --------------------------------------------------
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        # Phase 1: No disturbances, let it learn to walk first
        env = VelocityEnv(render_mode=None, enable_disturbance=False)
        env = TimeLimit(env, max_episode_steps=3000)
        env = Monitor(env)  # Record stats (reward, length)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # Windows/Linux safety check for multiprocessing
    multiprocessing.freeze_support()

    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"[INFO] Run Name: {RUN_NAME}")
    print(f"[INFO] CPUs detected: {multiprocessing.cpu_count()}")
    print(f"[INFO] Spawning {NUM_ENVS} parallel environments...")

    # --------------------------------------------------
    # 3. Create Parallel Environments
    # --------------------------------------------------
    # SubprocVecEnv runs each env in a separate CPU process
    env = SubprocVecEnv(
        [make_env(i, SEED) for i in range(NUM_ENVS)]
    )

    if RESUME:
        print(f"[INFO] Loading VecNormalize from {RESUME_VEC_PATH}")
        env = VecNormalize.load(RESUME_VEC_PATH, env)
        env.training = True
        env.norm_reward = True
    
    else:
        # Normalize Observations AND Rewards
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # --------------------------------------------------
    # 4. Evaluation Environment (Single Process)
    # --------------------------------------------------
    # We evaluate on just 1 env to keep logs clean
    eval_env = make_env(rank=99, seed=SEED)()
    eval_env = Monitor(eval_env)
    # Wrap manually since make_vec_env isn't used here for single instance
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Critical: Use a separate Normalizer for Eval that doesn't update!

    if RESUME:
        print(f"[INFO] Loading Eval VecNormalize from {RESUME_VEC_PATH}")
        eval_env = VecNormalize.load(RESUME_VEC_PATH, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        eval_env.training = False 
        eval_env.norm_reward = False

    # --------------------------------------------------
    # 5. Model Setup
    # --------------------------------------------------
    if RESUME:
        print(f"[INFO] Resuming PPO from {RESUME_MODEL_PATH}")
        model = PPO.load(
            RESUME_MODEL_PATH,
            env=env,
            device="cpu",
            tensorboard_log=LOG_DIR
        )
        print(f"[INFO] Model timestep counter = {model.num_timesteps}")

    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cpu",
            **HYPERPARAMS
        )

    # --------------------------------------------------
    # 6. Callbacks
    # --------------------------------------------------
    # Save every 500k steps (NOTE: 'n_calls' counts per-env, so we adjust freq)
    save_freq_steps = 200_000 
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq_steps // NUM_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_velocity"
    )

    vec_norm_callback = SaveVecNormalizeCallback(
        save_freq=max(save_freq_steps // NUM_ENVS, 1),
        save_path=CHECKPOINT_DIR
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(CHECKPOINT_DIR, "best"),
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=max(100_000 // NUM_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # --------------------------------------------------
    # 7. Start Training
    # --------------------------------------------------
    TOTAL_TIMESTEPS = 30_000_000
    
    print(f"[INFO] Starting training for {TOTAL_TIMESTEPS} timesteps...")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback, vec_norm_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted. Saving progress...")
    finally:
        # Final Save
        final_model_path = os.path.join(CHECKPOINT_DIR, "final_model")
        final_stats_path = os.path.join(CHECKPOINT_DIR, "vec_normalize.pkl")
        
        model.save(final_model_path)
        env.save(final_stats_path)
        
        print(f"[DONE] Model saved to: {final_model_path}.zip")
        print(f"[DONE] Stats saved to: {final_stats_path}")
        
        env.close()
        eval_env.close()