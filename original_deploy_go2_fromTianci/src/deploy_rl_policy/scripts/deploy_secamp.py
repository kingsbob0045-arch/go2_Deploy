#!/usr/bin/env python3
"""
deploy_secamp.py — ROS 2 node for the SECAMP motion prior on Unitree Go2.

Prior obs layout (45-dim, single frame, no history):
  [0:3]   projected_gravity
  [3:6]   [cos(θ)*0.5, sin(θ)*0.5, 0.0]   (body-frame direction command)
  [6:18]  (dof_pos - default) * dof_pos_scale   (Isaac order)
  [18:30] dof_vel * dof_vel_scale               (Isaac order)
  [30:42] last_action                            (Isaac order)
  [42:45] skill_onehot                           (0=pace, 1=trot, 2=canter)

Two modes (--mode):
  waypoint  (default) — follows DIR_WAYPOINTS time schedule
  joystick            — left stick drives direction; face buttons switch skill

Usage:
    python3 deploy_secamp.py --is_simulation True --mode waypoint
    python3 deploy_secamp.py --is_simulation True --mode joystick
"""

import math
import argparse
import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from pathlib import Path
from unitree_go.msg import LowState
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

project_root = Path(__file__).resolve().parents[3]

# MuJoCo XML (FR/FL/RR/RL) <-> Isaac Gym URDF (FL/FR/RL/RR)
# Self-inverse: REORDER[REORDER] == arange(12)
REORDER = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

SKILL_NAMES = ["pace", "trot", "canter"]

# Waypoint schedule: (start_time_sec, direction_deg, skill_idx)
# direction_deg: 0=forward, 90=left, -90=right
# skill_idx: 0=pace, 1=trot, 2=canter, -1=stop
# 3 segments (pace / trot / canter), each with 2 waypoints, matching training episode structure.
# Each skill runs ~16 s; wp1→wp2 transition at the midpoint of each segment.
# Direction stays 0° (straight) throughout — training used random directions, but
# a straight showcase is clearest for validation.
DIR_WAYPOINTS = [
    # --- segment 1: pace (skill=0) ---
    ( 0.0,  0.0,  0),   # pace  wp1 start
    ( 8.0,  0.0,  0),   # pace  wp2
    # --- segment 2: trot (skill=1) ---
    (16.0,  0.0,  1),   # trot  wp1 start
    (24.0,  0.0,  1),   # trot  wp2
    # --- segment 3: canter (skill=2) ---
    (32.0,  0.0,  2),   # canter wp1 start
    (40.0,  0.0,  2),   # canter wp2
    # --- stop ---
    (48.0,  0.0, -1),
]


class SecampConfig:
    def __init__(self, path):
        with open(path) as f:
            c = yaml.load(f, Loader=yaml.FullLoader)
        self.motion_prior_path = c["motion_prior_path"]
        self.default_angles    = np.array(c["default_angles"], dtype=np.float32)
        self.dof_pos_scale     = float(c["dof_pos_scale"])
        self.dof_vel_scale     = float(c["dof_vel_scale"])
        self.cmd_dir_scale     = float(c["cmd_dir_scale"])
        self.action_scale      = float(c["action_scale"])
        self.num_obs           = int(c["num_obs"])
        self.num_dof           = int(c["num_dof"])
        self.default_skill     = int(c["skill"])
        self.joystick          = c.get("joystick", {})


def _dir_obs(deg: float, scale: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r) * scale, math.sin(r) * scale, 0.0], dtype=np.float32)


class SecampPolicyNode(Node):
    def __init__(self, config: SecampConfig, args):
        super().__init__("secamp_policy")
        self.config = config
        self.args   = args

        self.get_logger().info(f"Loading motion prior: {config.motion_prior_path}")
        self.policy = torch.jit.load(config.motion_prior_path, map_location="cpu").eval()

        self.qj          = np.zeros(config.num_dof, dtype=np.float32)
        self.dqj         = np.zeros(config.num_dof, dtype=np.float32)
        self.last_action = np.zeros(config.num_dof, dtype=np.float32)
        self.low_state   = LowState()

        first_wp = DIR_WAYPOINTS[0]
        self.cmd_dir_obs  = _dir_obs(first_wp[1], config.cmd_dir_scale)
        self.skill_onehot = np.zeros(3, dtype=np.float32)
        self.skill_onehot[config.default_skill] = 1.0

        self._start_time   = self.get_clock().now()
        self._prev_buttons = []
        self._step_count   = 0

        topic = "/mujoco/lowstate" if args.simulation else "/lowstate"
        self.create_subscription(LowState, topic, self.low_state_callback, 10)
        self.get_logger().info(f"Subscribed to {topic}")

        if args.mode == "joystick":
            self.create_subscription(Joy, "/joy", self.joy_callback, 10)
            self.get_logger().info("Joystick mode: left stick → direction, face buttons → skill")
        else:
            self.get_logger().info("Waypoint mode: following DIR_WAYPOINTS schedule")

        self.target_pos_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self.create_timer(0.02, self.policy_step)

    def low_state_callback(self, msg: LowState):
        self.low_state = msg

    def joy_callback(self, msg: Joy):
        cfg    = self.config.joystick
        dz     = float(cfg.get("deadzone", 0.15))
        ax_x   = int(cfg.get("axis_dir_x", 1))
        ax_y   = int(cfg.get("axis_dir_y", 0))
        sign_x = float(cfg.get("sign_dir_x", -1))
        sign_y = float(cfg.get("sign_dir_y", -1))
        btn_lb = int(cfg.get("btn_lb", 4))
        btn_rb = int(cfg.get("btn_rb", 5))
        btn_ids = [int(cfg.get("btn_pace", 0)),
                   int(cfg.get("btn_trot", 1)),
                   int(cfg.get("btn_canter", 2))]

        axes    = msg.axes
        buttons = list(msg.buttons)
        n_btn   = len(buttons)

        enabled = (n_btn > max(btn_lb, btn_rb) and
                   bool(buttons[btn_lb]) and bool(buttons[btn_rb]))

        if not enabled:
            self.cmd_dir_obs[0] = 0.0
            self.cmd_dir_obs[1] = 0.0
        else:
            dx = (axes[ax_x] * sign_x) if len(axes) > ax_x and abs(axes[ax_x]) > dz else 0.0
            dy = (axes[ax_y] * sign_y) if len(axes) > ax_y and abs(axes[ax_y]) > dz else 0.0
            norm = math.sqrt(dx * dx + dy * dy)
            if dx < 0 or norm < 1e-6:
                self.cmd_dir_obs[0] = 0.0
                self.cmd_dir_obs[1] = 0.0
            else:
                self.cmd_dir_obs[0] = dx / norm * self.config.cmd_dir_scale
                self.cmd_dir_obs[1] = dy / norm * self.config.cmd_dir_scale

        if not self._prev_buttons:
            self._prev_buttons = [False] * n_btn
        elif len(self._prev_buttons) != n_btn:
            self._prev_buttons = [False] * n_btn

        for i, btn_id in enumerate(btn_ids):
            if btn_id < n_btn:
                pressed = bool(buttons[btn_id])
                if pressed and not self._prev_buttons[btn_id]:
                    self.skill_onehot[:] = 0.0
                    self.skill_onehot[i] = 1.0
                    self.get_logger().info(f"Skill → {SKILL_NAMES[i]}")
                self._prev_buttons[btn_id] = pressed

    def policy_step(self):
        if self.args.mode == "waypoint":
            elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
            for t, deg, sk in reversed(DIR_WAYPOINTS):
                if elapsed >= t:
                    if sk == -1:
                        self.cmd_dir_obs[:] = 0.0
                    else:
                        self.cmd_dir_obs[:] = _dir_obs(deg, self.config.cmd_dir_scale)
                        self.skill_onehot[:] = 0.0
                        self.skill_onehot[sk] = 1.0
                    break

        # Read joint state from LowState (robot order) and reorder to Isaac order
        qj_raw  = np.array([self.low_state.motor_state[i].q  for i in range(12)], dtype=np.float32)
        dqj_raw = np.array([self.low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)
        self.qj  = qj_raw[REORDER]
        self.dqj = dqj_raw[REORDER]

        quat    = self.low_state.imu_state.quaternion  # [w, x, y, z]
        gravity = self.get_gravity_orientation(quat)
        dof_pos = (self.qj  - self.config.default_angles) * self.config.dof_pos_scale
        dof_vel = self.dqj  * self.config.dof_vel_scale

        obs = np.concatenate([
            gravity,
            self.cmd_dir_obs,
            dof_pos,
            dof_vel,
            self.last_action,
            self.skill_onehot,
        ]).astype(np.float32)  # 45-dim

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.inference_mode():
            action_t = self.policy(obs_t)
        self.last_action = action_t.squeeze(0).numpy()  # Isaac order

        target_isaac = self.last_action * self.config.action_scale + self.config.default_angles
        target_robot = target_isaac[REORDER]  # Isaac → robot order for /rl/target_pos

        out = Float32MultiArray()
        out.data = target_robot.astype(np.float32).tolist()
        self.target_pos_pub.publish(out)

        self._step_count += 1
        if self._step_count % 50 == 0:  # print once per second
            norm = math.sqrt(self.cmd_dir_obs[0] ** 2 + self.cmd_dir_obs[1] ** 2)
            if norm < 1e-6:
                cmd_str = "no_cmd"
            else:
                cmd_deg = math.degrees(math.atan2(self.cmd_dir_obs[1], self.cmd_dir_obs[0]))
                cmd_str = f"{cmd_deg:+.1f}°"
            qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
            yaw_deg = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy),
                                              1.0 - 2.0 * (qy * qy + qz * qz)))
            gyro_z = math.degrees(self.low_state.imu_state.gyroscope[2])
            skill = SKILL_NAMES[int(np.argmax(self.skill_onehot))]
            print(f"cmd={cmd_str:<10}  yaw={yaw_deg:+.1f}°  yaw_rate={gyro_z:+.1f}°/s  skill={skill}")

    @staticmethod
    def get_gravity_orientation(quaternion) -> np.ndarray:
        qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        return np.array([
             2.0 * (-qz * qx + qw * qy),
            -2.0 * ( qz * qy + qw * qx),
             1.0 - 2.0 * (qw * qw + qz * qz),
        ], dtype=np.float32)


def main():
    rclpy.init()
    node = SecampPolicyNode(config=config, args=args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    config_path = project_root / "src" / "deploy_rl_policy" / "configs" / "go2_secamp.yaml"
    config = SecampConfig(config_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--is_simulation", choices=["True", "False"], default="True")
    parser.add_argument("--mode", choices=["waypoint", "joystick"], default="waypoint")
    args = parser.parse_args()
    args.simulation = args.is_simulation == "True"

    main()
