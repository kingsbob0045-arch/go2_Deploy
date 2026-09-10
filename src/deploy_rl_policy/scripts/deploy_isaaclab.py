#!/usr/bin/env python3
"""ROS 2 deployment for policies exported by the migrated Isaac Lab project."""

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowState

from xbox_command import XboxController

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Isaac FL/FR/RL/RR <-> Unitree/MuJoCo FR/FL/RR/RL. This permutation is self-inverse.
ISAAC_TO_UNITREE = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
EXPECTED_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
# Matches keyboard_command.py / xbox_command.py button indices.
BUTTON_TRIANGLE = 2
SKILL_NAMES = ("pace", "trot", "canter")


class DeployConfig:
    def __init__(self, path: Path):
        data = yaml.safe_load(path.read_text())
        for key, value in data.items():
            setattr(self, key, value)
        self.policy_path = self._resolve(self.policy_path)
        prior = getattr(self, "motion_prior_path", None)
        self.motion_prior_path = self._resolve(prior) if prior else None
        self.default_angles = np.asarray(self.default_angles, dtype=np.float32)
        self.command_scale = np.asarray(self.command_scale, dtype=np.float32)
        self.history_steps = int(getattr(self, "history_steps", 1))
        self.residual_action_scale = float(getattr(self, "residual_action_scale", 0.1))
        # Must match Go2AmpEnvCfg.action_clip in go2_RL_AMP; the training environment follows
        # legged_gym's clip_actions=100, so clipping tighter here would change the policy.
        self.action_clip = float(getattr(self, "action_clip", 100.0))

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


class IsaacLabPolicyNode(Node):
    def __init__(self, cfg: DeployConfig, simulation: bool):
        super().__init__("isaaclab_policy")
        self.cfg = cfg
        self._validate_contract()
        self.policy = torch.jit.load(str(cfg.policy_path), map_location="cpu").eval()
        self.motion_prior = None
        if cfg.policy_kind == "rough_residual":
            self.motion_prior = torch.jit.load(str(cfg.motion_prior_path), map_location="cpu").eval()

        self.controller = XboxController(self)
        self.low_state = None
        self.last_action = np.zeros(12, dtype=np.float32)
        self.history = np.zeros((cfg.history_steps, 42), dtype=np.float32)
        self.history_ready = False
        self.skill = np.zeros(3, dtype=np.float32)
        self.skill[int(getattr(cfg, "default_skill", 0))] = 1.0
        self._previous_buttons: list = []

        topic = "/mujoco/lowstate" if simulation else "/lowstate"
        self.create_subscription(LowState, topic, self._low_state_callback, 10)
        self.publisher = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self.create_timer(float(cfg.control_dt), self._policy_step)
        self.get_logger().info(f"Loaded Isaac Lab {cfg.policy_kind} policy: {cfg.policy_path}")
        self.get_logger().info(f"Reading {topic}; publishing Unitree FR/FL/RR/RL joint targets")

    def _validate_contract(self) -> None:
        if not self.cfg.policy_path.is_file():
            raise FileNotFoundError(f"Policy not found: {self.cfg.policy_path}")
        if self.cfg.default_angles.shape != (12,):
            raise ValueError("default_angles must contain 12 values in Isaac FL/FR/RL/RR order")
        if self.cfg.policy_kind == "rough_residual":
            if self.cfg.motion_prior_path is None or not self.cfg.motion_prior_path.is_file():
                raise FileNotFoundError(f"Motion prior not found: {self.cfg.motion_prior_path}")

        sidecar = self.cfg.policy_path.with_suffix(".json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing export sidecar {sidecar}; export with isaaclab/scripts/play.py")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("source") != "Isaac Lab":
            raise ValueError(f"{sidecar} is not an Isaac Lab policy contract")
        if metadata.get("joint_order") != EXPECTED_JOINT_ORDER:
            raise ValueError(f"Joint-order mismatch in {sidecar}")
        if metadata.get("policy_kind") != self.cfg.policy_kind:
            raise ValueError(
                f"policy_kind mismatch: config={self.cfg.policy_kind}, export={metadata.get('policy_kind')}"
            )
        expected_obs = 42 * self.cfg.history_steps + (3 if self.cfg.policy_kind == "secamp" else 0)
        expected_actions = 15 if self.cfg.policy_kind == "rough_residual" else 12
        if metadata.get("observation_dim") != expected_obs or metadata.get("action_dim") != expected_actions:
            raise ValueError(
                f"Policy dimensions mismatch: expected obs/action={expected_obs}/{expected_actions}, "
                f"export has {metadata.get('observation_dim')}/{metadata.get('action_dim')}"
            )

    def _low_state_callback(self, msg: LowState) -> None:
        self.low_state = msg

    def _policy_step(self) -> None:
        if self.low_state is None:
            return
        q_raw = np.asarray([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)
        dq_raw = np.asarray([self.low_state.motor_state[i].dq for i in range(12)], dtype=np.float32)
        q, dq = q_raw[ISAAC_TO_UNITREE], dq_raw[ISAAC_TO_UNITREE]
        gravity = self._projected_gravity(self.low_state.imu_state.quaternion)

        command = np.zeros(3, dtype=np.float32)
        left, right = self.controller.is_pressed()
        if left and right:
            command[:2] = self.controller.get_left_stick()
            command[2] = self.controller.get_right_stick()
        self._update_skill()
        base_obs = np.concatenate((
            gravity, command * self.cfg.command_scale,
            (q - self.cfg.default_angles) * float(self.cfg.dof_pos_scale),
            dq * float(self.cfg.dof_vel_scale), self.last_action,
        )).astype(np.float32)

        if not self.history_ready:
            self.history[:] = base_obs
            self.history_ready = True
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = base_obs
        if self.cfg.policy_kind == "secamp":
            policy_obs = np.concatenate((base_obs, self.skill))
        elif self.cfg.policy_kind == "rough_residual":
            policy_obs = self.history.reshape(-1)
        else:
            policy_obs = base_obs

        with torch.inference_mode():
            actor_output = self.policy(torch.from_numpy(policy_obs).unsqueeze(0)).squeeze(0)
        if self.cfg.policy_kind == "rough_residual":
            residual, skill = actor_output[:12], torch.softmax(actor_output[12:15], dim=-1)
            prior_obs = torch.from_numpy(base_obs.copy()).unsqueeze(0)
            xy = torch.from_numpy(command[:2]).unsqueeze(0)
            prior_obs[:, 3:5] = xy / torch.linalg.vector_norm(xy, dim=-1, keepdim=True).clamp(min=1.0e-6) * 0.5
            prior_obs[:, 5] = 0.0
            with torch.inference_mode():
                prior_action = self.motion_prior(torch.cat((prior_obs, skill.unsqueeze(0)), dim=-1)).squeeze(0)
            action = (prior_action + self.cfg.residual_action_scale * residual).clamp(
                -self.cfg.action_clip, self.cfg.action_clip)
        else:
            action = actor_output[:12].clamp(-self.cfg.action_clip, self.cfg.action_clip)

        self.last_action = action.cpu().numpy().astype(np.float32)
        target_isaac = self.cfg.default_angles + float(self.cfg.action_scale) * self.last_action
        message = Float32MultiArray()
        message.data = target_isaac[ISAAC_TO_UNITREE].tolist()
        self.publisher.publish(message)

    def _update_skill(self) -> None:
        """Cycle the SECAMP gait on a rising edge of Triangle (keyboard 'r').

        X and O are already claimed by low_level_ctrl for lay down / stand up, so Triangle
        is the only free pulse button in keyboard_command.py's map.
        """
        if self.cfg.policy_kind != "secamp":
            return
        buttons = list(self.controller.buttons)
        if len(buttons) > BUTTON_TRIANGLE and not self._previous_buttons:
            self._previous_buttons = buttons
            return
        if (len(buttons) > BUTTON_TRIANGLE and len(self._previous_buttons) > BUTTON_TRIANGLE
                and buttons[BUTTON_TRIANGLE] and not self._previous_buttons[BUTTON_TRIANGLE]):
            index = (int(self.skill.argmax()) + 1) % 3
            self.skill[:] = 0.0
            self.skill[index] = 1.0
            self.get_logger().info(f"skill -> {SKILL_NAMES[index]}")
        self._previous_buttons = buttons

    @staticmethod
    def _projected_gravity(quaternion) -> np.ndarray:
        qw, qx, qy, qz = quaternion
        return np.asarray([
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "src/deploy_rl_policy/configs/go2_isaaclab.yaml"))
    parser.add_argument("--is_simulation", choices=("True", "False"), default="True")
    args = parser.parse_args()
    rclpy.init()
    node = IsaacLabPolicyNode(DeployConfig(Path(args.config)), args.is_simulation == "True")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
