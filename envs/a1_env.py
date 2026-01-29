import numpy as np
import mujoco

from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box

from scipy.spatial.transform import Rotation as R



# ------------------------------------------------------------
# Camera configuration
# ------------------------------------------------------------
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,     # trunk
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 0.4)),
    "elevation": -20.0,
}


# ------------------------------------------------------------
# A1 Environment
# ------------------------------------------------------------
class A1Env(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        xml_file: str = "./models/unitree_a1/scene.xml",
        frame_skip: int = 5,
        render_mode: str | None = None,
        **kwargs,
    ):
        # EzPickle for Gymnasium reproducibility
        utils.EzPickle.__init__(self, xml_file, frame_skip, render_mode, **kwargs)

        # Initialize MuJoCo
        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,   # defined explicitly below
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )

        # ----------------------------------------------------
        # Control / timing
        # ----------------------------------------------------
        self.control_dt = self.dt * self.frame_skip


        # --- Base ID ---
        self.trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        # ----------------------------------------------------
        # Actuated joint definitions (source of truth)
        # ----------------------------------------------------
        self.JOINT_NAMES = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]

        self.num_joints = len(self.JOINT_NAMES)
        assert self.num_joints == 12

        # ----------------------------------------------------
        # Map joint names → qpos / qvel indices
        # ----------------------------------------------------
        self.joint_qpos_ids = []
        self.joint_qvel_ids = []

        for name in self.JOINT_NAMES:
            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            assert jid != -1, f"Joint {name} not found in model"

            self.joint_qpos_ids.append(self.model.jnt_qposadr[jid])
            self.joint_qvel_ids.append(self.model.jnt_dofadr[jid])

        self.joint_qpos_ids = np.array(self.joint_qpos_ids)
        self.joint_qvel_ids = np.array(self.joint_qvel_ids)

        # ----------------------------------------------------
        # Nominal pose (anchor for position control)
        # ----------------------------------------------------
        self.nominal_qpos = self.init_qpos[self.joint_qpos_ids].copy()

        # ----------------------------------------------------
        # Action space (normalized position offsets)
        # ----------------------------------------------------
        self.action_scale = 0.15  # radians

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Observation space (fixed, placeholder for now)
        # ----------------------------------------------------
        self.obs_dim = 47  # will be filled properly next
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Commands
        self.v_cmd = 0.0
        self.yaw_rate_cmd = 0.0

        self.v_cmd_max = 2.0
        self.yaw_rate_cmd_max = 1.0

        self.foot_body_names = [
            "FR_foot",
            "FL_foot",
            "RR_foot",
            "RL_foot",
        ]

        self.foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.foot_body_names
        ]

        self.allowed_foot_geoms = set()

        for foot_body in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, foot_body
            )

            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == body_id:
                    self.allowed_foot_geoms.add(g)



        self.ground_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor"
        )

        # --- Disturbance config ---
        self.enable_disturbance = True

        self.DIST_PROB = 0.02
        self.DIST_FORCE = 80.0
        self.DIST_DURATION = 10

        self.disturb_steps_left = 0
        self.disturb_dir = np.zeros(3)

        self.T_SWING_TARGET = 0.15   # seconds
        self.T_SWING_MAX = 0.35
        self.swing_reward_scale = 0.3
        self.swing_over_penalty = 0.5

        assert all(bid != -1 for bid in self.foot_body_ids)



    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------
    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        self.data.xfrc_applied[:] = 0.0
        self.disturb_steps_left = 0

        # Orientation noise
        roll  = self.np_random.uniform(-0.2, 0.2)
        pitch = self.np_random.uniform(-0.2, 0.2)
        yaw   = self.np_random.uniform(-0.2, 0.2)
        qpos[3:7] = self.euler_to_quat(roll, pitch, yaw)

        # Joint noise
        qpos[self.joint_qpos_ids] += self.np_random.uniform(
            -0.05, 0.05, size=self.num_joints
        )

        self.set_state(qpos, qvel)

        # 🔑 Sample velocity command
        self.v_cmd = self.np_random.uniform(0.5, 1.5)
        # self.yaw_rate_cmd = self.np_random.uniform(-0.5, 0.5)
        self.yaw_rate_cmd = 0.0

        self.feet_air_time = np.zeros(4)
        self.last_contacts = np.zeros(4, dtype=bool)

        self.last_action = np.zeros(self.num_joints)

        return self._get_obs()



    # --------------------------------------------------------
    # Step
    # --------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        alpha = np.clip(abs(self.v_cmd) / 0.5, 0.0, 1.0)

        q_ref = (1 - alpha) * self.nominal_qpos + alpha * self.data.qpos[self.joint_qpos_ids]
        qpos_target = q_ref + self.action_scale * action

        # Apply disturbance BEFORE stepping physics
        self._apply_disturbance()

        # Let Gymnasium handle ctrl assignment
        self.do_simulation(qpos_target, self.frame_skip)

        observation = self._get_obs()
        reward = self._compute_reward()

        illegal_contact = self._has_illegal_contact()

        terminated = self._check_termination() or illegal_contact
        truncated = False

        if self.render_mode == "human":
            self.render()

        self.last_action = action.copy()

        return observation, reward, terminated, truncated, {}


    # --------------------------------------------------------
    # Observation (stub)
    # --------------------------------------------------------
    def _get_obs(self):
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]

        proj_grav = self._get_projected_gravity()

        cmd = np.array([
            self.v_cmd / self.v_cmd_max,
            self.yaw_rate_cmd / self.yaw_rate_cmd_max,
        ])

        q = self.data.qpos[self.joint_qpos_ids]
        qd = self.data.qvel[self.joint_qvel_ids]

        q_rel = q - self.nominal_qpos

        obs = np.concatenate([
            base_lin_vel,
            base_ang_vel,
            proj_grav,
            cmd,
            q_rel,
            qd,
            self.last_action,
        ])

        assert obs.shape == (self.obs_dim,), f"Obs shape mismatch: {obs.shape}"

        return obs.astype(np.float32)

    # --------------------------------------------------------
    # Reward (stub)
    # --------------------------------------------------------
    def _compute_reward(self):
        v_forward = self._get_base_forward_velocity()
        v_err = v_forward - self.v_cmd

        # --- main objective ---
        r_vel = np.exp(-2.0 * v_err**2)

        # --- stability ---
        proj_grav = self._get_projected_gravity()
        r_upright = np.exp(-2.0 * np.sum(proj_grav[:2]**2))

        # --- smoothness ---
        qd = self.data.qvel[self.joint_qvel_ids]
        r_smooth = np.exp(-0.1 * np.mean(qd**2))

        # --- posture only when slow ---
        posture_weight = np.exp(-3.0 * abs(self.v_cmd))
        q = self.data.qpos[self.joint_qpos_ids]
        r_posture = posture_weight * np.exp(-np.mean((q - self.nominal_qpos)**2))

        reward = (
            2.0 * r_vel +
            0.5 * r_upright +
            0.3 * r_smooth
            # 0.2 * r_posture
        )

        contacts = self._get_foot_contacts()
        first_contact = (~self.last_contacts) & contacts

        # accumulate air time
        self.feet_air_time += self.control_dt
        self.feet_air_time[contacts] = 0.0
        
        # ----------------------------------------
        # Air-time timing reward (non-exploitable)
        # ----------------------------------------
        v = abs(self.v_cmd)
        T = np.clip(0.35 - 0.08 * v, 0.18, 0.35)

        for i in range(4):
            if first_contact[i]:
                swing_time = self.feet_air_time[i]

                # reward landing near target swing time
                reward += self.swing_reward_scale * np.exp(
                    -10.0 * (swing_time - T) ** 2
                )

                # penalize excessive swing duration
                if swing_time > self.T_SWING_MAX:
                    reward -= self.swing_over_penalty * (swing_time - self.T_SWING_MAX)


        # reward only when stepping and moving
        # air_time_reward = np.sum(
        #     (self.feet_air_time - 0.2) * first_contact
        # )

        base_height = self.data.qpos[2]
        height_error = base_height - 0.27  # nominal A1 height

        # r_height = np.exp(-20.0 * height_error**2)

        v_z = self.data.qvel[2]

        # print(f"base_height: {v_forward}")
        # air_time_reward *= abs(self.v_cmd) > 0.1

        self.last_contacts = contacts

        pitch_rate = self.data.qvel[4]  # pitch angular velocity
        r_pitch_stability = np.exp(-5.0 * pitch_rate ** 2)
        
        contacts = self._get_foot_contacts()
        rear_contacts = contacts[2:]
        front_contacts = contacts[:2]

        # print(f"[ENV] f_v: {v_forward:.3f} v_cmd: {self.v_cmd:.3f}")

        # if np.sum(rear_contacts) == 2 and np.sum(front_contacts) == 0:
        #     reward -= 0.2

        # reward += 0.05 * air_time_reward

        # reward += 0.3 * r_height
        reward -= 3.0 * height_error ** 2
        reward += 0.3 * np.tanh(5.0 * (base_height - 0.24))
        reward += 0.1 * r_pitch_stability
        reward -= 0.5 * v_z**2

        if abs(self.v_cmd) > 0.2:
            reward += 0.05 * np.mean(np.abs(qd))

        return reward



    # --------------------------------------------------------
    # Termination (stub)
    # --------------------------------------------------------
    def _check_termination(self):

        proj_grav = self._get_projected_gravity()
        return np.linalg.norm(proj_grav[:2]) > 0.8
        
    def _get_base_forward_velocity(self):
        # World linear velocity of base
        v_world = self.data.qvel[0:3]

        # Base orientation
        base_quat = self.data.qpos[3:7]
        rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])

        # Rotate world velocity into body frame
        v_body = rot.inv().apply(v_world)

        return v_body[0]  # forward (x)
    

    def quat2euler(self, quat):
            

        rot = R.from_quat([
            quat[1],  # x
            quat[2],  # y
            quat[3],  # z
            quat[0],  # w
        ])

        roll, pitch, yaw = rot.as_euler("xyz", degrees=False)

        return roll, pitch, yaw
    
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
    
    def _get_projected_gravity(self):
        """
        Gravity vector expressed in base frame.
        This replaces roll/pitch everywhere.
        """
        gravity_world = np.array(self.model.opt.gravity)  # [0, 0, -9.81]
        quat = self.data.qpos[3:7]  # w, x, y, z

        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        gravity_body = rot.inv().apply(gravity_world)

        return gravity_body / (np.linalg.norm(gravity_body) + 1e-8)

    def _get_foot_contacts(self, threshold=1.0):
        """
        Returns boolean array of foot contacts based on external forces.
        """
        forces = np.linalg.norm(
            self.data.cfrc_ext[self.foot_body_ids], axis=1
        )
        return forces > threshold
    

    def _has_illegal_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)

            body1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[g1]
            )
            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[g2]
            )

            # now apply legality rule
            if g1 == self.ground_geom_id:
                other = g2
            elif g2 == self.ground_geom_id:
                other = g1
            else:
                continue

            if other not in self.allowed_foot_geoms:
                return True

        return False

    def _apply_disturbance(self):
        # Always clear previous forces
        self.data.xfrc_applied[self.trunk_body_id] = 0.0

        if not self.enable_disturbance:
            return

        # Possibly start a new disturbance
        if self.disturb_steps_left == 0:
            if self.np_random.random() < self.DIST_PROB:
                angle = self.np_random.uniform(0, 2 * np.pi)
                self.disturb_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
                # self.disturb_dir = np.array([1.0, 0.0, 0.0])
                self.disturb_steps_left = self.DIST_DURATION

        # Apply disturbance if active
        if self.disturb_steps_left > 0:
            self.data.xfrc_applied[self.trunk_body_id, :3] = (
                self.DIST_FORCE * self.disturb_dir
            )
            self.disturb_steps_left -= 1
