import time

import numpy as np
import onnxruntime as rt
from etils import epath
from keyboard_reader import KeyboardController
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from utils import (
    RESTRICTED_JOINT_RANGE,
    Go2_NUM_MOTOR,
    Kd,
    Kp,
    NonBlockingInput,
    PosStopF,
    VelStopF,
    action_scale,
    default_pos,
    get_gravity_orientation,
    joint2motor_idx,
    rest_pos,
)

NETWORK_CARD_NAME = "enp47s0"
_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE


class OnnxPolicy:
    """ONNX controller for the Go-2 robot."""

    def __init__(
        self,
        policy_path: str,
    ):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CUDAExecutionProvider"])

    def get_control(self, obs: np.ndarray) -> None:
        onnx_input = {"obs": obs.reshape(1, -1)}
        onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
        return onnx_pred


class Controller:
    def __init__(self, policy: OnnxPolicy) -> None:
        self.policy = policy

        self.qj = np.zeros(Go2_NUM_MOTOR, dtype=np.float32)
        self.dqj = np.zeros(Go2_NUM_MOTOR, dtype=np.float32)
        self.action = np.zeros(Go2_NUM_MOTOR, dtype=np.float32)
        self.counter = 0

        # Convert joint range tuples to numpy arrays for efficient clamping
        joint_limits = np.array(RESTRICTED_JOINT_RANGE, dtype=np.float32)
        self._joint_lower_bounds = joint_limits[:, 0]
        self._joint_upper_bounds = joint_limits[:, 1]

        self._controller = KeyboardController(
            vel_scale_x=1.5,
            vel_scale_y=0.8,
            vel_scale_rot=np.pi / 4,
        )

        self.control_dt = 0.02

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        print("passed motion switch")
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        self.default_pos_array = np.array(default_pos)

        # wait for the subscriber to receive data
        self.wait_for_low_state()
        self.state_record = []
        self.ctrl_record = []

        # Initialize the command msg
        self.init_cmd_low_level()
        self.crc = CRC()

    def send_cmd(self, cmd: LowCmd_):
        cmd.crc = self.crc.Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Press Enter to continue...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key("\n"):
                self.create_zero_cmd()
                self.send_cmd(self.low_cmd)
        print("Zero torque state confirmed. Proceeding...")

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def init_cmd_low_level(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q = PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def create_damping_cmd(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 5
            self.low_cmd.motor_cmd[i].tau = 0

    # Send Zero torque command
    def create_zero_cmd(self):
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def move_to_joint_pos(self, end_pos, total_time=2):
        print("Moving to default pos.")
        # move time 2s
        num_step = int(total_time / self.control_dt)

        # Init_dof_pos foot: FL FR RL RR
        # low_state.motor_state: FR FL RR RL
        init_dof_pos = np.zeros(Go2_NUM_MOTOR, dtype=np.float32)
        for i in range(Go2_NUM_MOTOR):
            init_dof_pos[i] = self.low_state.motor_state[joint2motor_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            state_i = np.zeros(12)
            ctrl_i = np.zeros(12)
            for j in range(Go2_NUM_MOTOR):
                motor_idx = joint2motor_idx[j]
                target_pos = end_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = Kp[j]
                self.low_cmd.motor_cmd[motor_idx].kd = Kd[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

                ctrl_i[j] = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                state_i[j] = self.low_state.motor_state[motor_idx].q
            self.send_cmd(self.low_cmd)

            self.state_record.append(state_i)
            self.ctrl_record.append(ctrl_i)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Press Enter to start the controller...")
        with NonBlockingInput() as nbi:
            while not nbi.check_key("\n"):  # Check for Enter key
                # Keep sending default position commands while waiting
                for i in range(len(joint2motor_idx)):
                    motor_idx = joint2motor_idx[i]
                    self.low_cmd.motor_cmd[motor_idx].q = default_pos[i]
                    self.low_cmd.motor_cmd[motor_idx].dq = 0
                    self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
                    self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
                    self.low_cmd.motor_cmd[motor_idx].tau = 0

                self.send_cmd(self.low_cmd)
                time.sleep(self.control_dt)
        print("Default position state confirmed. Starting controller...")

    def run(self):
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(Go2_NUM_MOTOR):
            self.qj[i] = self.low_state.motor_state[joint2motor_idx[i]].q - default_pos[i]
            self.dqj[i] = self.low_state.motor_state[joint2motor_idx[i]].dq

        quat = self.low_state.imu_state.quaternion
        gyro = self.low_state.imu_state.gyroscope
        # lin_vel is not available

        # create observation
        gravity = get_gravity_orientation(quat)
        joint_angles = self.qj.copy()
        joint_velocities = self.dqj.copy()
        command = self._controller.get_command()  # Original line
        obs = np.hstack(
            [
                # lin_vel,
                gyro,
                gravity,
                joint_angles,
                joint_velocities,
                self.action,
                command,
            ]
        ).astype(np.float32)

        self.action = self.policy.get_control(obs)
        # print("Action: ", self.action)
        action_effect = self.action * action_scale

        motor_targets_unclamped = self.default_pos_array + action_effect

        # Clamp motor targets to joint limits and check for clamping
        motor_targets = np.clip(motor_targets_unclamped, self._joint_lower_bounds, self._joint_upper_bounds)
        clamped_indices = np.where(motor_targets != motor_targets_unclamped)[0]
        if clamped_indices.size > 0:
            print("WARNING: Clamping motor targets for joints:")
            for idx in clamped_indices:
                print(
                    f"  Joint {idx}: {motor_targets_unclamped[idx]:.3f} -> {motor_targets[idx]:.3f} (limits: [{self._joint_lower_bounds[idx]:.3f}, {self._joint_upper_bounds[idx]:.3f}])"
                )

        # Build low cmd
        for i in range(Go2_NUM_MOTOR):
            motor_idx = joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = motor_targets[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = Kp[i]
            self.low_cmd.motor_cmd[motor_idx].kd = Kd[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.control_dt)


if __name__ == "__main__":
    print("Setting up policy...")
    policy = OnnxPolicy((_ONNX_DIR / "go2.onnx").as_posix())
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    # Initial prompt doesn't need non-blocking
    input("Press Enter to acknowledge warning and proceed...")

    ChannelFactoryInitialize(0, NETWORK_CARD_NAME)

    controller = Controller(policy)

    # Initial prompt doesn't need non-blocking
    # input("Press Enter to acknowledge warning and proceed...")

    # Enter the zero torque state, press Enter key to continue executing
    controller.zero_torque_state()

    # Move to reset position
    controller.move_to_joint_pos(rest_pos, 1)

    # Move to the default position
    controller.move_to_joint_pos(default_pos, 2)

    # Enter the default position state, press Enter key to continue executing
    controller.default_pos_state()

    print("Controller running. Press 'p' to quit.")
    with NonBlockingInput() as nbi:  # Use context manager for the main loop
        while True:
            controller.run()
            # Check for 'q' key press to exit
            if nbi.check_key("p"):
                print("\n'p' pressed. Exiting loop...")
                break
            # Add a small sleep to prevent busy-waiting if controller.run() is very fast
            time.sleep(0.001)

    print("Entering damping state...")
    controller.create_damping_cmd()
    controller.send_cmd(controller.low_cmd)

    np.savetxt("state_record.txt", controller.state_record)
    np.savetxt("ctrl_record.txt", controller.ctrl_record)
    print("Exit")
