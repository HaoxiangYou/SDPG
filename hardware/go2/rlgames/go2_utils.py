import select
import sys
import termios
import tty

import numpy as np

Go2_NUM_MOTOR = 12

Kp = [
    35.0,
    35.0,
    35.0,
    35.0,
    35.0,
    35.0,  # legs
    35.0,
    35.0,
    35.0,
    35.0,
    35.0,
    35.0,  # legs
]

Kd = [
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,  # legs
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,  # legs
]

rest_pos = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0.2, 1.36, -2.65, -0.2, 1.36, -2.65]

default_pos = [
    0.0,
    0.8,
    -1.5,  # FR
    0.0,
    0.8,
    -1.5,  # FL
    0.0,
    1.0,
    -1.5,  # RR
    0.0,
    1.0,
    -1.5,  # RL
]

PosStopF = 2.146e9
VelStopF = 16000.0
lin_vel_scale = 2.0
action_scale = 0.5
ang_vel_scale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.05


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


# --- Non-blocking Keyboard Input Context Manager ---
class NonBlockingInput:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        try:
            tty.setraw(sys.stdin.fileno())
        except termios.error as e:
            # Fallback if not a tty (e.g., running in certain IDEs/environments)
            print(f"Warning: Could not set raw mode: {e}. Key detection might not work.", file=sys.stderr)
            self.old_settings = None  # Indicate failure
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        print("\nRestored terminal settings.")  # Optional: provide feedback

    def check_key(self, key="\n"):
        """Check if a specific key is pressed without blocking."""
        if not self.old_settings:  # If raw mode failed, don't check
            return False
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ch = sys.stdin.read(1)
            # In raw mode, Enter is often '\r' (carriage return)
            return ch == (key if key != "\n" else "\r")
        return False


# -----------------------------------------------------

# Go2 returns FR FL RR RL
joint2motor_idx = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]

RESTRICTED_JOINT_RANGE = (
    (-0.5, 0.5),  # FR_hip_joint
    (-1.4, 2.07),  # FR_thigh_joint
    (-2.0, -1.0),  # FR_calf_joint
    (-0.5, 0.5),  # FL_hip_joint
    (-1.4, 2.07),  # FL_thigh_joint
    (-2.0, -1.0),  # FL_calf_joint
    (-0.5, 0.5),  # RR_hip_joint
    (0.4, 2.9),  # RR_thigh_joint
    (-2.6, -1.4),  # RR_calf_joint
    (-0.5, 0.5),  # RL_hip_joint
    (0.4, 2.9),  # RL_thigh_joint
    (-2.6, -1.4),  # RL_calf_joint
)
