#!/usr/bin/env python3
"""End-to-end check: keyboard keys -> /joy -> low_level_ctrl state machine.

Runs a stub robot in place of MuJoCo so the real C++ state machine can be driven
headlessly, then replays the q/e/c key sequence and asserts the resulting state
transitions.  Start low_level_ctrl separately:

    ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true
"""

import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray
from unitree_go.msg import LowCmd, LowState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from keyboard_command import KeyboardJoystick  # noqa: E402

LAYDOWN = [-0.0325, 1.2635, -2.8057, 0.0508, 1.2487, -2.8048,
           -0.3644, 1.2937, -2.8195, 0.3402, 1.2838, -2.8085]
STANDING = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]


class StubRobot(Node):
    """Follow the commanded joint targets instantly so state_obs() sees clean states."""

    def __init__(self):
        super().__init__("stub_robot")
        self.q = list(LAYDOWN)
        self.create_subscription(LowCmd, "/mujoco/lowcmd", self.on_cmd, 10)
        self.state_pub = self.create_publisher(LowState, "/mujoco/lowstate", 10)
        self.joy_pub = self.create_publisher(Joy, "/joy", 10)
        self.target_pos_pub = self.create_publisher(Float32MultiArray, "/rl/target_pos", 10)
        self.create_timer(0.002, self.publish_state)
        self.controller = KeyboardJoystick()
        self.create_timer(0.02, self.publish_joy)
        self.policy_target_seen = False
        self.create_subscription(Float32MultiArray, "/pos", self.on_pos, 10)
        self.last_pos = None

    def on_cmd(self, msg: LowCmd) -> None:
        # Perfect tracking keeps get_norm(dq) at 0 so state_obs() settles immediately.
        self.q = [float(msg.motor_cmd[i].q) for i in range(12)]

    def on_pos(self, msg: Float32MultiArray) -> None:
        self.last_pos = list(msg.data)

    def publish_state(self) -> None:
        message = LowState()
        for i in range(12):
            message.motor_state[i].q = self.q[i]
            message.motor_state[i].dq = 0.0
        message.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.state_pub.publish(message)

    def publish_joy(self) -> None:
        self.controller.update()
        message = Joy()
        message.header.stamp = self.get_clock().now().to_msg()
        message.axes = [float(v) for v in self.controller.axes]
        message.buttons = [int(v) for v in self.controller.buttons]
        self.joy_pub.publish(message)

    def press(self, key: str) -> None:
        self.controller.handle_key(key)

    def distance_to(self, target) -> float:
        return sum((a - b) ** 2 for a, b in zip(self.q, target)) ** 0.5


def spin_for(node: Node, seconds: float) -> None:
    """The executor runs in a background thread, so just let wall-clock time pass."""
    time.sleep(seconds)


def main() -> int:
    rclpy.init()
    node = StubRobot()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name} {detail}")
        if not condition:
            failures.append(name)

    def wait_until(predicate, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    print("\nWaiting for low_level_ctrl to drive the stub into the laydown pose...")
    reached = wait_until(lambda: node.distance_to(LAYDOWN) < 0.25, 15.0)
    check("initial state is laydown", reached, f"(distance {node.distance_to(LAYDOWN):.3f})")

    print("\nPressing 'e' (stand up)...")
    node.press("e")
    reached = wait_until(lambda: node.distance_to(STANDING) < 0.3, 15.0)
    check("e reaches the standing pose", reached, f"(distance {node.distance_to(STANDING):.3f})")

    print("\nPressing 'c' (latch L1+R1) then publishing a policy target...")
    node.press("c")
    node.controller.update()
    check("latch holds buttons 4 and 5",
          node.controller.buttons[4] == 1 and node.controller.buttons[5] == 1)
    probe = [value + 0.05 for value in STANDING]
    for _ in range(40):
        message = Float32MultiArray()
        message.data = [float(v) for v in probe]
        node.target_pos_pub.publish(message)
        time.sleep(0.02)
    check("policy targets reach the robot while latched", node.distance_to(probe) < 0.05,
          f"(distance {node.distance_to(probe):.3f})")

    print("\nPressing 'c' again (release the latch)...")
    node.press("c")
    node.controller.update()
    check("second c releases buttons 4 and 5",
          node.controller.buttons[4] == 0 and node.controller.buttons[5] == 0)

    print("\nPressing 'q' (lay down)...")
    node.press("q")
    reached = wait_until(lambda: node.distance_to(LAYDOWN) < 0.25, 20.0)
    check("q returns to the laydown pose", reached, f"(distance {node.distance_to(LAYDOWN):.3f})")

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
