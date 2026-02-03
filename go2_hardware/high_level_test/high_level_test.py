import pickle
import sys
import threading
import time

import numpy as np
from pynput import keyboard
from pynput.keyboard import KeyCode
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
)
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_


class Custom:
    def __init__(self):
        self.traj_path = "scalemmd_trajectory.pkl"
        self.current_pose = np.zeros(7)

    # Public methods
    def Init(self):
        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.lidar_subscriber = ChannelSubscriber("rt/utlidar/robot_odom", Odometry_)
        self.lidar_subscriber.Init(self.LidarMessageHandler, 10)

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        self.low_state = None

        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.highstate_subscriber.Init(self.HighStateHandler, 10)
        self.high_state = None

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.pos_world = np.zeros(3)
        self.quat_world = np.zeros(4)

        self.vel_scale_x = 0.3
        self.vel_scale_y = 0.3
        self.vel_scale_rot = 1.0
        self.tracking_vel = 0.3

        self.key_combo = {
            KeyCode.from_char("a"),  # left
            KeyCode.from_char("w"),  # forward
            KeyCode.from_char("s"),  # backward
            KeyCode.from_char("d"),  # right
            KeyCode.from_char("q"),  # rot left
            KeyCode.from_char("e"),  # rot right
            KeyCode.from_char("p"),
        }  # stop

        self.currently_pressed = set()
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.stop = False

        with open(self.traj_path, "rb") as file:
            self.traj_points = pickle.load(file)
        self.target_point_idx = 0

    def on_press(self, key):
        """Handle key press events."""
        self.currently_pressed.add(key)
        self.update_velocity()

    def on_release(self, key):
        """Handle key release events."""
        # Update velocity when key is released
        if key == KeyCode.from_char("w") or key == KeyCode.from_char("s"):
            self.vx = 0.0
        elif key == KeyCode.from_char("a") or key == KeyCode.from_char("d"):
            self.vy = 0.0
        elif key == KeyCode.from_char("q") or key == KeyCode.from_char("e"):
            self.vyaw = 0.0
        elif key == KeyCode.from_char("p"):
            self.stop = True
            return False  # Stop listener

        self.currently_pressed.discard(key)
        self.update_velocity()

    def update_velocity(self):
        """Update velocity based on currently pressed keys."""
        # Reset velocities
        vx = 0.0
        vy = 0.0
        vyaw = 0.0

        # Check for key presses
        if KeyCode.from_char("w") in self.currently_pressed:
            vx = self.vel_scale_x  # Forward
        if KeyCode.from_char("s") in self.currently_pressed:
            vx = -self.vel_scale_x  # Backward
        if KeyCode.from_char("a") in self.currently_pressed:
            vy = self.vel_scale_y  # Strafe left
        if KeyCode.from_char("d") in self.currently_pressed:
            vy = -self.vel_scale_y  # Strafe right
        if KeyCode.from_char("q") in self.currently_pressed:
            vyaw = self.vel_scale_rot  # Rotate left
        if KeyCode.from_char("e") in self.currently_pressed:
            vyaw = -self.vel_scale_rot  # Rotate right

        # Update velocities
        self.vx = vx
        self.vy = vy
        self.vyaw = vyaw

    def Start(self):
        self.sport_client.StandUp()
        input("Press Enter to unlock joint...")
        self.sport_client.BalanceStand()
        # turn on obstacle avoidance
        self.sport_client.FreeAvoid(True)
        stop_event = threading.Event()

        print("\n" + "=" * 60)
        print("Go2 Keyboard Control")
        print("=" * 60)
        print("Controls:")
        print("  W - Move forward")
        print("  S - Move backward")
        print("  A - Strafe left")
        print("  D - Strafe right")
        print("  Q - Rotate left")
        print("  E - Rotate right")
        print("  P - Stop and exit")
        print("=" * 60)
        print("\nMake sure the terminal window has focus to receive keyboard input.")

        with True:
            input("Press Enter to start keyboard control...")
            print("\nRobot ready! Use WASD keys to control movement.")
            print("Press P to stop and exit.\n")

            while True:
                # Send movement command using sport_client.Move()
                self.sport_client.Move(self.vx, self.vy, self.vyaw)

                time.sleep(0.05)  # 20 Hz control loop

                if self.stop:
                    self.sport_client.StopMove()
                    time.sleep(0.5)
                    self.sport_client.StandDown()
                    time.sleep(2)
                    self.sport_client.Damp()
                    stop_event.set()
                    self.listener.join()
                    break

    def HighStateHandler(self, msg: SportModeState_):
        self.high_state = msg

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

    def LidarMessageHandler(self, msg: Odometry_):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.current_pose[0] = position.x
        self.current_pose[1] = position.y
        self.current_pose[2] = position.z
        self.current_pose[3] = orientation.w
        self.current_pose[4] = orientation.x
        self.current_pose[5] = orientation.y
        self.current_pose[6] = orientation.z


if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()
