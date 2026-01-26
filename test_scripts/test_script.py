import mujoco
import mujoco.viewer
import time
import numpy as np

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = mujoco.MjModel.from_xml_path("./models/balance_v1/scene.xml")
data = mujoco.MjData(model)
dt = model.opt.timestep

# --------------------------------------------------
# IDs
# --------------------------------------------------
imu_gyro_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
)
gyro_adr = model.sensor_adr[imu_gyro_id]

base_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base_chassis"
)

yaw_jid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw"
)
yaw_qvel_adr = model.jnt_dofadr[yaw_jid]

# --------------------------------------------------
# Viewer
# --------------------------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    # Let the model settle
    for _ in range(100):
        mujoco.mj_step(model, data)

    while viewer.is_running():

        # ------------------------------------------
        # Apply constant yaw torque (world Z)
        # ------------------------------------------
        data.xfrc_applied[base_body_id, 3:6] = [0.0, 0.0, -0.3]

        mujoco.mj_step(model, data)

        # ------------------------------------------
        # Read sensors
        # ------------------------------------------
        gx, gy, gz = data.sensordata[gyro_adr : gyro_adr + 3]
        yaw_rate_joint = data.qvel[yaw_qvel_adr]

        print(
            f"gyro_z: {gz:+.4f} rad/s | "
            f"yaw_joint_vel: {yaw_rate_joint:+.4f} rad/s"
        )

        viewer.sync()
        time.sleep(dt)
