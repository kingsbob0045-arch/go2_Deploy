#!/usr/bin/env python3
"""keyboard_command.py - publish sensor_msgs/Joy on /joy from the keyboard.

Drop-in replacement for `ros2 run joy joy_node` when no gamepad is available.
Nothing downstream changes: low_level_ctrl.cpp, deploy_secamp.py, rl_policy.py
and mujoco_simulator.py keep reading the same /joy contract.

Gamepad contract reproduced here (see docs/gameController.txt):
    buttons[0] X         laydown   (also skill pace   in deploy_secamp.py)
    buttons[1] O         stand up  (also skill trot   in deploy_secamp.py)
    buttons[2] Triangle            (also skill canter in deploy_secamp.py)
    buttons[4] L1 + buttons[5] R1  held together: activate policy / enable sticks
    axes[0] left  stick horizontal (lateral)
    axes[1] left  stick vertical   (forward)
    axes[3] right stick horizontal (yaw)
    axes[4] right stick vertical
    axes[2] L2, axes[5] R2         both at -1.0 asks low_level_ctrl to shut down

Keyboard map:
    q            laydown            -> buttons[0] pulse
    e            stand up           -> buttons[1] pulse
    r            canter skill       -> buttons[2] pulse
    c            toggle L1+R1 latch -> buttons[4] and buttons[5] held/released
    w / s        left  stick  +/- forward   -> axes[1]
    a / d        left  stick  +/- lateral   -> axes[0]
    up / down    right stick  +/- vertical  -> axes[4]
    left / right right stick  +/- yaw       -> axes[3]
    space        centre every stick
    x            emulate L2+R2, asks low_level_ctrl to shut down
    ? or h       reprint this map
    Ctrl-C / Esc quit this node only

Stick modes (--stick-mode):
    latch  (default) a direction key stays applied until the same key is pressed
                     again, the opposite key is pressed, or space centres it.
                     Independent of terminal key-repeat, so it always works.
    hold             a direction key applies while the terminal auto-repeats it
                     and decays to zero --hold-timeout seconds after the last
                     repeat.  Closer to a real spring-return stick, but it
                     needs keyboard auto-repeat to be enabled.

Usage:
    python3 keyboard_command.py                     # latch mode
    python3 keyboard_command.py --stick-mode hold
    ros2 run deploy_rl_policy keyboard_command.py
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# A real controller reports six axes and at least eight buttons; keep the same
# shape so index-based consumers never fall into their "len(axes) > n" guards.
NUM_AXES = 6
NUM_BUTTONS = 8

# Released analogue triggers rest at +1.0 and read -1.0 when fully pressed.
TRIGGER_RELEASED = 1.0
TRIGGER_PRESSED = -1.0
AXIS_LEFT_X, AXIS_LEFT_Y = 0, 1
AXIS_RIGHT_X, AXIS_RIGHT_Y = 3, 4
AXIS_L2, AXIS_R2 = 2, 5
BUTTON_X, BUTTON_O, BUTTON_TRIANGLE = 0, 1, 2
BUTTON_L1, BUTTON_R1 = 4, 5

# key -> (axis index, value)
STICK_KEYS = {
    "w": (AXIS_LEFT_Y, 1.0),
    "s": (AXIS_LEFT_Y, -1.0),
    "a": (AXIS_LEFT_X, 1.0),
    "d": (AXIS_LEFT_X, -1.0),
    "up": (AXIS_RIGHT_Y, 1.0),
    "down": (AXIS_RIGHT_Y, -1.0),
    "left": (AXIS_RIGHT_X, 1.0),
    "right": (AXIS_RIGHT_X, -1.0),
}

# key -> button index fired as a short pulse
PULSE_KEYS = {
    "q": BUTTON_X,
    "e": BUTTON_O,
    "r": BUTTON_TRIANGLE,
}

HELP_TEXT = """
  keyboard -> /joy
  ---------------------------------------------------------------
   q            lay down            (X)
   e            stand up            (O)
   r            canter skill        (Triangle)
   c            toggle L1+R1 latch  -> activates the RL policy
   w / s        forward  / backward (left stick)
   a / d        left     / right    (left stick)
   arrows       right stick (yaw on left-right)
   space        centre every stick
   x            L2+R2, tells low_level_ctrl to shut down
   ? or h       show this map again
   Ctrl-C       quit this node
  ---------------------------------------------------------------
"""


class KeyboardJoystick:
    """Turn key presses into a gamepad-shaped (axes, buttons) state.

    Free of ROS and Isaac Lab imports so the exact same logic can be reused by
    isaaclab/scripts/play.py in go2_RL_AMP.
    """

    def __init__(self, stick_mode: str = "latch", hold_timeout: float = 0.6,
                 pulse_duration: float = 0.15):
        self.stick_mode = stick_mode
        self.hold_timeout = hold_timeout
        self.pulse_duration = pulse_duration
        self.axes = [0.0] * NUM_AXES
        self.axes[AXIS_L2] = TRIGGER_RELEASED
        self.axes[AXIS_R2] = TRIGGER_RELEASED
        self.buttons = [0] * NUM_BUTTONS
        self.latched = False
        self.shutdown_requested = False
        self._pulse_until: dict[int, float] = {}
        self._stick_seen: dict[str, float] = {}
        self._messages: list[str] = []

    def drain_messages(self) -> list[str]:
        messages, self._messages = self._messages, []
        return messages

    def handle_key(self, key: str) -> None:
        now = time.monotonic()
        if key in PULSE_KEYS:
            button = PULSE_KEYS[key]
            self.buttons[button] = 1
            self._pulse_until[button] = now + self.pulse_duration
            self._messages.append(f"button {button} pressed ({key})")
        elif key == "c":
            self.latched = not self.latched
            self._messages.append(
                "L1+R1 LATCHED - policy active, sticks live" if self.latched
                else "L1+R1 released - sticks ignored")
        elif key in STICK_KEYS:
            axis, value = STICK_KEYS[key]
            if self.stick_mode == "hold":
                self.axes[axis] = value
                self._stick_seen[key] = now
            elif self.axes[axis] == value:
                self.axes[axis] = 0.0  # same key again centres this axis
            else:
                self.axes[axis] = value
        elif key == "space":
            for axis in (AXIS_LEFT_X, AXIS_LEFT_Y, AXIS_RIGHT_X, AXIS_RIGHT_Y):
                self.axes[axis] = 0.0
            self._stick_seen.clear()
            self._messages.append("sticks centred")
        elif key == "x":
            self.shutdown_requested = True
            self._messages.append("L2+R2 - asking low_level_ctrl to shut down")
        elif key in ("?", "h"):
            self._messages.append(HELP_TEXT)

    def update(self) -> None:
        """Expire button pulses, decay held sticks, and refresh latched buttons."""
        now = time.monotonic()
        for button, deadline in list(self._pulse_until.items()):
            if now >= deadline:
                self.buttons[button] = 0
                del self._pulse_until[button]

        if self.stick_mode == "hold":
            for key, last_seen in list(self._stick_seen.items()):
                if now - last_seen > self.hold_timeout:
                    axis, value = STICK_KEYS[key]
                    if self.axes[axis] == value:
                        self.axes[axis] = 0.0
                    del self._stick_seen[key]

        held = 1 if self.latched else 0
        self.buttons[BUTTON_L1] = held
        self.buttons[BUTTON_R1] = held
        trigger = TRIGGER_PRESSED if self.shutdown_requested else TRIGGER_RELEASED
        self.axes[AXIS_L2] = trigger
        self.axes[AXIS_R2] = trigger

    def status_line(self) -> str:
        return (f"latch={'ON ' if self.latched else 'off'}  "
                f"L[x={self.axes[AXIS_LEFT_X]:+.0f} y={self.axes[AXIS_LEFT_Y]:+.0f}]  "
                f"R[x={self.axes[AXIS_RIGHT_X]:+.0f} y={self.axes[AXIS_RIGHT_Y]:+.0f}]")


class RawKeyboard:
    """Read single key presses from a terminal without waiting for Enter."""

    ESCAPE_SEQUENCES = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}

    def __init__(self, stream=None):
        self._stream = stream or sys.stdin
        self._fd = self._stream.fileno()
        self._saved = None

    def __enter__(self):
        if not os.isatty(self._fd):
            raise RuntimeError(
                "keyboard_command needs an interactive terminal. Run it directly in a "
                "terminal rather than through a pipe, a launch file, or nohup.")
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_exc_info):
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def poll(self) -> list[str]:
        """Return every key name buffered right now; never blocks."""
        keys = []
        while select.select([self._fd], [], [], 0)[0]:
            char = os.read(self._fd, 1).decode("utf-8", errors="ignore")
            if not char:
                break
            if char == "\x1b":  # arrow keys arrive as ESC [ A..D
                sequence = ""
                while len(sequence) < 2 and select.select([self._fd], [], [], 0.005)[0]:
                    sequence += os.read(self._fd, 1).decode("utf-8", errors="ignore")
                keys.append(self.ESCAPE_SEQUENCES.get(sequence, "esc"))
            elif char == " ":
                keys.append("space")
            elif char == "\x03":
                raise KeyboardInterrupt
            else:
                keys.append(char.lower())
        return keys


class KeyboardJoyNode(Node):
    def __init__(self, controller: KeyboardJoystick, keyboard: RawKeyboard, rate_hz: float):
        super().__init__("keyboard_command")
        self.controller = controller
        self.keyboard = keyboard
        self.publisher = self.create_publisher(Joy, "/joy", 10)
        self.create_timer(1.0 / rate_hz, self.tick)
        self._ticks = 0
        self._status_every = max(1, int(rate_hz))
        self.get_logger().info(f"Publishing /joy at {rate_hz:.0f} Hz "
                               f"(stick mode: {controller.stick_mode})")
        print(HELP_TEXT, flush=True)

    def tick(self) -> None:
        for key in self.keyboard.poll():
            if key == "esc":
                raise KeyboardInterrupt
            self.controller.handle_key(key)
        self.controller.update()

        for message in self.controller.drain_messages():
            print(message, flush=True)

        message = Joy()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "keyboard"
        message.axes = [float(value) for value in self.controller.axes]
        message.buttons = [int(value) for value in self.controller.buttons]
        self.publisher.publish(message)

        self._ticks += 1
        if self._ticks % self._status_every == 0:
            print(f"\r{self.controller.status_line()}   ", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stick-mode", choices=("latch", "hold"), default="latch")
    parser.add_argument("--hold-timeout", type=float, default=0.6,
                        help="Seconds without auto-repeat before a held stick centres")
    parser.add_argument("--rate", type=float, default=50.0, help="/joy publish rate in Hz")
    # ROS 2 appends --ros-args; ignore anything this parser does not define.
    args, _ = parser.parse_known_args()

    rclpy.init()
    controller = KeyboardJoystick(stick_mode=args.stick_mode, hold_timeout=args.hold_timeout)
    try:
        with RawKeyboard() as keyboard:
            node = KeyboardJoyNode(controller, keyboard, args.rate)
            try:
                rclpy.spin(node)
            except KeyboardInterrupt:
                pass
            finally:
                node.destroy_node()
    finally:
        print()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
