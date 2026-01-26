import numpy as np
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import math


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 0.5)),
    "elevation": -20.0,
}

class VelocityEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, xml_file="./balance_v1.1/scene.xml", frame_skip=4, enable_disturbance=False, **kwargs):
        
        self.enable_disturbance = enable_disturbance

        # --- Constants ---
        self.WHEEL_RADIUS = 0.06
        self.MAX_TILT = 0.8  
        self.v_cmd_scale = 1.5
        self.v_cmd = 0.0

        self.yaw_cmd = 0.0
        self.yaw_cmd_scale = 2.0
        self.max_wheel_speed = 30 # rad/s

        self.MAX_WHEEL_TORQUE = 10.0
        self.MAX_HIP_STEP = 0.03
        self.MAX_KNEE_STEP = 0.03

        self.HIP_MIN, self.HIP_MAX = -0.4, 0.4
        self.KNEE_MIN, self.KNEE_MAX = 0.0, 0.5

        self.BASE_Z_MIN = 0.141
        self.BASE_Z_MAX = 0.37
        # self.MIN_WHEEL_SPEED = 0.01 # rad/s


        # --- Disturbance ---
        self.DIST_FORCE = 5.0
        self.DIST_DURATION_CTRL = 20
        self.DIST_PROB = 0.0004
        self.disturb_steps_left = 0
        self.disturb_dir = np.zeros(2)

        # --- State Estimation ---
        self.theta_est = 0.0
        self.roll_est = 0.0

        utils.EzPickle.__init__(self, xml_file, frame_skip, enable_disturbance, **kwargs)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # self.physics_dt = self.model.opt.timestep
        self.control_dt = self.model.opt.timestep * self.frame_skip

        # Complementary filter time constant (seconds)
        self.tau = 0.3

        self.n_actuators = self.model.nu 
        self.last_action = np.zeros(self.n_actuators, dtype=np.float64)

        # Obs: 6 Sensors + 4 Joint Angles + 4 velocity of joints + 6 Last Actions,
        # NOTE: It's no 20 in total. last time I removed the motor actuators for hips and knees
        # obs_dim = 8 + 4 + 4 + self.n_actuators 
        obs_dim = (
            11 +    # core body + command + wheel states
            4 +     # joint positions
            4 +     # joint velocities
            self.n_actuators
        )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        # --- Cache IDs ---
        self.imu_acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        self.imu_gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        self.acc_adr = self.model.sensor_adr[self.imu_acc_id]
        self.gyro_adr = self.model.sensor_adr[self.imu_gyro_id]
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_chassis")

        lw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        rw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self.lw_dof = self.model.jnt_dofadr[lw]
        self.rw_dof = self.model.jnt_dofadr[rw]

        # --- Explicit Mapping (UPDATED to match XML) ---
        self.actuator_order = [
            "left_wheel_joint",   # type: motor
            "right_wheel_joint",  # type: motor
            "left_hip_pos_con",   # type: position
            "right_hip_pos_con",  # type: position
            "left_knee_pos_con",  # type: position
            "right_knee_pos_con"  # type: position
        ]
        
        self.act_ids = {}
        for name in self.actuator_order:
            # We look for mjOBJ_ACTUATOR, not JOINT
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if id == -1: 
                raise ValueError(f"Actuator '{name}' not found! Check XML names.")
            self.act_ids[name] = id

        # --- Joint IDs and addresses for observation ---
        self.joint_names = [
            "left_hip_joint",
            "right_hip_joint",
            "left_knee_joint",
            "right_knee_joint",
        ]

        self.joint_qpos_adrs = []
        self.joint_qvel_adrs = []

        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")

            self.joint_qpos_adrs.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_adrs.append(self.model.jnt_dofadr[jid])

        # Symmetry regularization
        self.W_SYM = 0.1
        

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        prev_action = self.last_action.copy()

        # --- Disturbance logic (control-rate, frame-skip safe) ---
        if self.enable_disturbance:

            if self.disturb_steps_left == 0 and self.np_random.random() < self.DIST_PROB:
                angle = self.np_random.uniform(0, 2 * np.pi)
                self.disturb_dir = np.array([np.cos(angle), np.sin(angle)])
                self.disturb_steps_left = self.DIST_DURATION_CTRL

            if self.disturb_steps_left > 0:
                self.data.xfrc_applied[self.base_body_id, :2] = (
                    self.DIST_FORCE * self.disturb_dir
                )
                self.disturb_steps_left -= 1
            else:
                self.data.xfrc_applied[self.base_body_id, :2] = 0.0

                        
        # Wheels (torque)
        self.data.ctrl[self.act_ids["left_wheel_joint"]]  = action[0] * self.MAX_WHEEL_TORQUE
        self.data.ctrl[self.act_ids["right_wheel_joint"]] = action[1] * self.MAX_WHEEL_TORQUE

        # Hips (absolute position targets)
        hip_L = self.HIP_MIN + (action[2] + 1) * 0.5 * (self.HIP_MAX - self.HIP_MIN)
        hip_R = self.HIP_MIN + (action[3] + 1) * 0.5 * (self.HIP_MAX - self.HIP_MIN)

        knee_L = self.KNEE_MIN + (action[4] + 1) * 0.5 * (self.KNEE_MAX - self.KNEE_MIN)
        knee_R = self.KNEE_MIN + (action[5] + 1) * 0.5 * (self.KNEE_MAX - self.KNEE_MIN)

        self.data.ctrl[self.act_ids["left_hip_pos_con"]]  = hip_L
        self.data.ctrl[self.act_ids["right_hip_pos_con"]] = hip_R
        self.data.ctrl[self.act_ids["left_knee_pos_con"]] = knee_L
        self.data.ctrl[self.act_ids["right_knee_pos_con"]] = knee_R
        

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        obs = self._get_obs()
        reward = self._compute_reward(action, prev_action)
        terminated = self._check_done() # "Immortal" logic
        
        self.last_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}
    
    def reset_model(self):
        # 1. Prepare the Skeleton (qpos/qvel) in Python variables
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # --- Randomize Base ---
        pitch = self.np_random.uniform(-0.2, 0.2)
        roll = self.np_random.uniform(-0.15, 0.15)
        yaw = 0.0
        qpos[3:7] = self.euler_to_quat(roll, pitch, yaw)   
        qpos[0] += self.np_random.uniform(-0.05, 0.05)
        qpos[1] += self.np_random.uniform(-0.02, 0.02)

        # --- Randomize Legs ---
        hip_L = self.np_random.uniform(-0.4, 0.4)
        hip_R = self.np_random.uniform(-0.4, 0.4)
        knee_L = self.np_random.uniform(0.0, 0.5)
        knee_R = self.np_random.uniform(0.0, 0.5)

        # Helper to find qpos address from joint name
        def get_adr(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[jid]

        # Set qpos values directly
        qpos[get_adr("left_hip_joint")] = hip_L
        qpos[get_adr("right_hip_joint")] = hip_R
        qpos[get_adr("left_knee_joint")] = knee_L
        qpos[get_adr("right_knee_joint")] = knee_R

        # # 2. Prepare the Brain (ctrl) directly
        # # Since we haven't touched the physics engine yet, we can just write to data.ctrl
        self.data.ctrl[self.act_ids["left_hip_pos_con"]] = hip_L
        self.data.ctrl[self.act_ids["right_hip_pos_con"]] = hip_R
        self.data.ctrl[self.act_ids["left_knee_pos_con"]] = knee_L
        self.data.ctrl[self.act_ids["right_knee_pos_con"]] = knee_R

        # 3. Apply Everything to Physics ONE TIME
        # This pushes qpos/qvel to the engine AND runs forward dynamics 
        # considering the 'ctrl' values we just set.
        self.set_state(qpos, qvel)

        # 4. Initialize Internal Variables
        acc = self.data.sensordata[self.acc_adr:self.acc_adr + 3]

        # self.theta_est = self.accel_pitch(*acc)
        # self.roll_est = self.accel_roll(*acc)
        ax, ay, az = acc
        g = math.sqrt(ax*ax + ay*ay + az*az) + 1e-6
        ax_n, ay_n, az_n = ax/g, ay/g, az/g

        self.theta_est = self.accel_pitch(ax_n, ay_n, az_n)
        self.roll_est  = self.accel_roll(ax_n, ay_n, az_n)

        self.last_action[:] = 0.0

        # self.v_cmd = 0.0
        self.v_cmd = self.np_random.uniform(-1.5, 1.5)

        if abs(self.v_cmd) < 0.15:
            self.v_cmd = 0.15 * np.sign(self.v_cmd) if self.v_cmd != 0 else 0.15

        # max_yaw = max(0.6, abs(self.v_cmd))
        self.yaw_cmd = self.np_random.uniform(-0.8, 0.8)

        # Remove tiny yaw noise, but allow straight motion
        if abs(self.yaw_cmd) < 0.2:
            self.yaw_cmd = 0.0

        self.disturb_steps_left = 0

        return self._get_obs()

    def _get_obs(self):

        acc = self.data.sensordata[self.acc_adr:self.acc_adr + 3]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3]
        
        ax, ay, az = acc

        # Normalize accelerometer to remove magnitude scaling
        g = math.sqrt(ax*ax + ay*ay + az*az) + 1e-6
        ax_n, ay_n, az_n = ax/g, ay/g, az/g

        accel_pitch = self.accel_pitch(ax_n, ay_n, az_n)

        # self.theta_est = 0.98 * (self.theta_est + gyro[1] * self.physics_dt ) + 0.02 * self.accel_pitch(*acc)
        # self.roll_est = 0.98 * (self.roll_est + gyro[0] * self.physics_dt ) + 0.02 * self.accel_roll(*acc)

        dt = self.control_dt
        alpha = self.tau / (self.tau + dt)

        self.theta_est = (
            alpha * (self.theta_est + gyro[1] * dt)
            + (1.0 - alpha) * accel_pitch
        )

        self.roll_est = (
            alpha * (self.roll_est + gyro[0] * dt)
            + (1.0 - alpha) * self.accel_roll(ax_n, ay_n, az_n)
        )

        v_norm = self.forward_velocity() / self.v_cmd_scale
        v_cmd_norm = self.v_cmd / self.v_cmd_scale

        y_rate = self.data.sensordata[self.gyro_adr + 2] # z-axis gyro
        
        yaw_rate_norm = y_rate / self.yaw_cmd_scale
        yaw_cmd_norm = self.yaw_cmd / self.yaw_cmd_scale

        wl = self.data.qvel[self.lw_dof]
        wr = self.data.qvel[self.rw_dof]

        wl_norm = np.clip(wl / self.max_wheel_speed, -1.5, 1.5)
        wr_norm = np.clip(wr / self.max_wheel_speed, -1.5, 1.5)

        wheel_diff_norm = np.clip((wr - wl) / self.max_wheel_speed, -1.5, 1.5)

        # return np.concatenate([
        #     np.array([
        #         self.theta_est,
        #         gyro[1],
        #         self.roll_est,
        #         gyro[0],
        #         v_norm,
        #         v_cmd_norm,
        #         self.data.ctrl[self.act_ids["left_hip_pos_con"]],
        #         self.data.ctrl[self.act_ids["right_hip_pos_con"]],
        #         self.data.ctrl[self.act_ids["left_knee_pos_con"]],
        #         self.data.ctrl[self.act_ids["right_knee_pos_con"]],
        #     ], dtype=np.float64),
        #     self.last_action
        # ])

        return np.concatenate([
            np.array([
                self.theta_est,
                gyro[1],
                self.roll_est,
                gyro[0],
                v_norm,
                v_cmd_norm,
                yaw_rate_norm,
                yaw_cmd_norm,
                wl_norm,
                wr_norm,
                wheel_diff_norm,
                self.data.qpos[self.joint_qpos_adrs[0]],
                self.data.qpos[self.joint_qpos_adrs[1]],
                self.data.qpos[self.joint_qpos_adrs[2]],
                self.data.qpos[self.joint_qpos_adrs[3]],

                self.data.qvel[self.joint_qvel_adrs[0]],
                self.data.qvel[self.joint_qvel_adrs[1]],
                self.data.qvel[self.joint_qvel_adrs[2]],
                self.data.qvel[self.joint_qvel_adrs[3]],

            ], dtype=np.float64),
            self.last_action
        ])
    

    def _compute_reward(self, action, prev_action):
        theta = self.theta_est
        theta_dot = self.data.sensordata[self.gyro_adr + 1]
        roll = self.roll_est
        roll_dot = self.data.sensordata[self.gyro_adr]

        v_norm = self.forward_velocity() / self.v_cmd_scale
        v_cmd_norm = self.v_cmd / self.v_cmd_scale

        v_error = v_norm - v_cmd_norm

        # --- Velocity tracking reward ---
        # exp_velocity_reward = np.exp(-0.5 * v_error**2)
        # v_scale = max(0.2, abs(self.v_cmd))
        v_scale = 0.2 + 0.5 * abs(self.v_cmd)
        
        exp_velocity_reward = np.exp(-0.5 * (v_error / v_scale)**2)

        # upright = np.exp(-4.0 * theta**2) * np.exp(-2.0 * roll**2)
        upright = np.exp(-4.0 * theta**2)
        velocity_weight = 2.0 + 4.0 * upright


        yaw_rate = self.data.sensordata[self.gyro_adr + 2] # z-axis gyro
        
        yaw_rate_norm = yaw_rate / self.yaw_cmd_scale
        yaw_cmd_norm = self.yaw_cmd / self.yaw_cmd_scale

        yaw_error = yaw_rate_norm - yaw_cmd_norm

        # yaw_scale = max(0.2, abs(self.yaw_cmd) / self.yaw_cmd_scale)
        yaw_scale = 0.2 + 0.3 * abs(yaw_cmd_norm)

        yaw_reward = np.exp(-0.5 * (yaw_error / yaw_scale)**2)

        # yaw_active = np.exp(-0.3 * (yaw_cmd_norm ** 2))
        # yaw_active = 1.0 - yaw_active

        # --- Wheel difference shaping for yaw
        wl = self.data.qvel[self.lw_dof]
        wr = self.data.qvel[self.rw_dof]

        wheel_diff_norm = (wr - wl) / self.max_wheel_speed

        # Desired wheel difference comes directly from yaw command
        # desired_wheel_diff = yaw_cmd_norm * (self.yaw_cmd_scale / self.max_wheel_speed)
        desired_wheel_diff = yaw_cmd_norm

        wheel_diff_error = wheel_diff_norm - desired_wheel_diff

        wheel_diff_reward = np.exp(-2.0 * wheel_diff_error**2)


        # yaw_active = np.clip(abs(yaw_cmd_norm) / 0.2, 0.0, 1.0)
        yaw_active = np.clip(abs(self.yaw_cmd) / 0.6, 0.0, 1.0)

        yaw_quality = np.exp(-2.0 * yaw_error**2)

        velocity_weight *= (1.0 - 0.6 * yaw_active * (1.0 - yaw_quality))
        # velocity_weight *= (1.0 - 0.5 * yaw_active)

        # --------------------------------------------------
        # Fix 1: Joint symmetry penalty (STATE-based)
        # --------------------------------------------------
        q_l_hip  = self.data.qpos[self.joint_qpos_adrs[0]]
        q_r_hip  = self.data.qpos[self.joint_qpos_adrs[1]]
        q_l_knee = self.data.qpos[self.joint_qpos_adrs[2]]
        q_r_knee = self.data.qpos[self.joint_qpos_adrs[3]]

        sym_penalty = (
            (q_l_hip  - q_r_hip)**2 +
            (q_l_knee - q_r_knee)**2
        )

        sym_weight = self.W_SYM * (1.0 - yaw_active)

        ctrl_cost = 0.02 * np.sum(action[:2]**2)   # wheels
        ctrl_cost += 0.01 * np.sum(action[2:]**2)  # legs

        smoothness_cost = 0.01 * np.sum((action - prev_action)**2)
        
        roll_weight = 1.0 * (1.0 - 0.7 * yaw_active)
        reward = (
            3.0

            - 2.0 * theta**2
            - 1.2 * theta_dot ** 2
            - roll_weight * roll ** 2
            - 0.8 * roll_dot ** 2

            # + 1.5 * upright_bonus
            # + 2.0 * exp_velocity_reward
            + velocity_weight * exp_velocity_reward

            - ctrl_cost
            - smoothness_cost
        )

        # reward -= sym_weight * sym_penalty

        reward += yaw_active * (
            3.0 * yaw_reward +
            1.0 * wheel_diff_reward
        )

        reward -= yaw_active * (1.0 - yaw_reward)

        # deleted.
        # reward += yaw_active * 0.2 * yaw_torque_reward


        # To avoid twitchy turning:
        yaw_ctrl_cost = (1.0 - yaw_active) * 0.02 * (action[0] - action[1])**2
        reward -= yaw_ctrl_cost

        return reward
    
    def _check_done(self):
        if self.data.qpos[2] < 0.139: 
            return True
        
        # if abs(self.theta_est) > self.MAX_TILT:
        #     return True
        
        if not np.isfinite(self.state_vector()).all():
            return True
        return False

    def forward_velocity(self):
        wl = self.data.qvel[self.lw_dof]
        wr = self.data.qvel[self.rw_dof]
        return self.WHEEL_RADIUS * 0.5 * (wl + wr)

    def accel_pitch(self, ax, ay, az):
        return math.atan2(-ax, math.sqrt(ay * ay + az * az))

    def accel_roll(self, ax, ay, az):
        return math.atan2(ay, az)
    
    def euler_to_quat(self, roll, pitch, yaw):
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])
    

    def trigger_disturbance(self, force_xy, duration_steps):
        """
        External API for evaluation.
        Applies a constant force for `duration_steps` control steps.
        """
        self.disturb_dir = np.array(force_xy, dtype=np.float64)
        norm = np.linalg.norm(self.disturb_dir)
        if norm > 1e-6:
            self.disturb_dir /= norm
        self.DIST_FORCE = norm
        self.disturb_steps_left = duration_steps
