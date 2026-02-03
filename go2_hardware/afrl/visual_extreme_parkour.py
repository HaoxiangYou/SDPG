import json
import os
import os.path as osp
import sys
import time

import numpy as np
import pyrealsense2 as rs
import rclpy
import ros2_numpy as rnp
import torch
import torch.nn.functional as F
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from torch.autograd import Variable

if os.uname().machine in ["x86_64", "amd64"]:
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "x86",
        )
    )
elif os.uname().machine == "aarch64":
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "aarch64",
        )
    )


@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data


class VisualHandlerNode(Node):
    """A wapper class for the realsense camera"""

    def __init__(
        self,
        cfg: dict,
        cropping: tuple = (0, 0, 0, 0),  # top, bottom, left, right
        rs_resolution: tuple = (480, 270),  # width, height for the realsense camera)
        rs_fps: int = 30,
        depth_input_topic="/camera/forward_depth",
        camera_info_topic="/camera/camera_info",
        forward_depth_image_topic="/forward_depth_image",
    ):
        super().__init__("depth_image")
        self.cfg = cfg
        self.cropping = cropping
        self.rs_resolution = rs_resolution
        self.rs_fps = rs_fps
        self.depth_input_topic = depth_input_topic
        self.camera_info_topic = camera_info_topic
        self.forward_depth_image_topic = forward_depth_image_topic

        self.parse_args()
        self.start_pipeline()
        self.start_ros_handlers()

        # debug
        # depth_data_sim = np.load('/home/unitree/Desktop/extreme_parkour_onboard/depth_image_random.npy')
        # depth_data_sim = np.load('/home/unitree/Desktop/extreme_parkour_onboard/depth_image_sim_336-11_flat.npy')
        # self.depth_data_sim = torch.from_numpy(depth_data_sim.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def parse_args(self):
        # self.output_resolution = self.cfg["depth"]["resized"]
        self.output_resolution = [58, 87]
        depth_range = [0.0, 3.0]
        # depth_range = [0.0, 2.0]
        # self.depth_range = depth_range
        self.depth_range = (depth_range[0], depth_range[1] * 1000)  # [m] -> [mm]

    def start_pipeline(self):
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.depth,
            self.rs_resolution[0],
            self.rs_resolution[1],
            rs.format.z16,
            self.rs_fps,
        )
        self.rs_profile = self.rs_pipeline.start(self.rs_config)

        self.rs_align = rs.align(rs.stream.depth)

        # build rs builtin filters
        # self.rs_decimation_filter = rs.decimation_filter()
        # self.rs_decimation_filter.set_option(rs.option.filter_magnitude, 6)
        self.rs_hole_filling_filter = rs.hole_filling_filter()
        self.rs_spatial_filter = rs.spatial_filter()
        self.rs_spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
        self.rs_spatial_filter.set_option(rs.option.holes_fill, 4)
        self.rs_temporal_filter = rs.temporal_filter()
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
        self.rs_temporal_filter.set_option(rs.option.filter_smooth_delta, 1)
        # using a list of filters to define the filtering order
        self.rs_filters = [
            # self.rs_decimation_filter,
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]

    def start_ros_handlers(self):
        self.depth_input_pub = self.create_publisher(
            Image,
            self.depth_input_topic,
            1,
        )

        self.forward_depth_image_pub = self.create_publisher(
            Float32MultiArray,
            self.forward_depth_image_topic,
            1,
        )

        self.get_logger().info("ros handlers started")

    def publish_camera_info_callback(self):
        self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info("camera info published", once=True)
        self.camera_info_pub.publish(self.camera_info_msg)

    def get_depth_frame(self):
        # read from pyrealsense2, preprocess and write the model embedding to the buffer
        latency_range = [0.08, 0.142]
        rs_frame = self.rs_pipeline.wait_for_frames(int(latency_range[1] * 1000))  # ms

        depth_frame = rs_frame.get_depth_frame()
        if not depth_frame:
            self.get_logger().error("No depth frame", throttle_duration_sec=1)
            return

        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)

        depth_image_pyt = torch.from_numpy(np.asanyarray(depth_frame.get_data()).astype(np.float32)).unsqueeze(0)
        # depth_image_np = np.rot90(depth_image_np, k= 2) # k = 2 for rotate 90 degree twice

        # apply torch filters
        depth_image_pyt = depth_image_pyt[
            :,
            self.cropping[0] : -self.cropping[1] - 1,
            self.cropping[2] : -self.cropping[3] - 1,
        ]

        depth_image_pyt = torch.clip(depth_image_pyt, self.depth_range[0], self.depth_range[1]) / (
            self.depth_range[1] - self.depth_range[0]
        )
        depth_image_pyt = resize2d(depth_image_pyt, self.output_resolution)

        # publish the depth image input to ros topic
        self.get_logger().info("depth range: {}-{}".format(*self.depth_range), once=True)
        depth_input_data = (
            depth_image_pyt.detach().cpu().numpy() * (self.depth_range[1] - self.depth_range[0]) + self.depth_range[0]
        ).astype(np.uint16)[0]  # (h, w) unit [mm]
        # print('depth input data: ', depth_input_data.min(), depth_input_data.max())

        depth_image_pyt -= 0.5  # [-0.5, 0.5])

        depth_input_msg = rnp.msgify(Image, depth_input_data.astype(np.float32), encoding="32FC1")
        depth_input_msg.header.stamp = self.get_clock().now().to_msg()
        depth_input_msg.header.frame_id = "d435_sim_depth_link"
        self.depth_input_pub.publish(depth_input_msg)
        self.get_logger().info("depth input published", once=True)

        return depth_image_pyt

    def publish_depth_data(self, depth_data):
        msg = Float32MultiArray()
        msg.data = depth_data.flatten().detach().cpu().numpy().tolist()

        self.forward_depth_image_pub.publish(msg)
        self.get_logger().info("depth data published", once=True)

    def start_main_loop_timer(self, duration):
        self.create_timer(
            duration,
            self.main_loop,
        )

    def main_loop(self):
        depth_image_pyt = self.get_depth_frame()
        if depth_image_pyt is not None:
            # depth_image_pyt = torch.zeros_like(depth_image_pyt) + 0.5
            # depth_image_pyt = self.depth_data_sim
            self.publish_depth_data(depth_image_pyt)
        else:
            self.get_logger().warn("One frame of depth latent if not acquired")


@torch.inference_mode()
def main(args):
    rclpy.init()

    # duration = config_dict["sensor"]["forward_camera"]["refresh_duration"] # in sec
    duration = 0.01  # duration

    visual_node = VisualHandlerNode(
        cfg=json.load(open(osp.join(args.logdir, "config.json"), "r")),
        cropping=[args.crop_top, args.crop_bottom, args.crop_left, args.crop_right],
        rs_resolution=(args.width, args.height),
        rs_fps=args.fps,
    )

    if args.loop_mode == "while":
        rclpy.spin_once(visual_node, timeout_sec=0.0)
        while rclpy.ok():
            main_loop_time = time.monotonic()
            visual_node.main_loop()
            rclpy.spin_once(visual_node, timeout_sec=0.0)
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
    elif args.loop_mode == "timer":
        visual_node.start_main_loop_timer(duration)
        rclpy.spin(visual_node)

    visual_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="The height of the realsense image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="The width of the realsense image",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="The fps request to the rs pipeline",
    )
    parser.add_argument(
        "--crop_left",
        type=int,
        default=80,  # 28
        help="num of pixel to crop in the original pyrealsense readings.",
    )
    parser.add_argument(
        "--crop_right",
        type=int,
        default=36,  # 36
        help="num of pixel to crop in the original pyrealsense readings.",
    )
    parser.add_argument(
        "--crop_top",
        type=int,
        default=60,  # 48
        help="num of pixel to crop in the original pyrealsense readings.",
    )
    parser.add_argument(
        "--crop_bottom", type=int, default=100, help="num of pixel to crop in the original pyrealsense readings."
    )
    parser.add_argument(
        "--loop_mode",
        type=str,
        default="timer",
        choices=["while", "timer"],
        help="Select which mode to run the main policy control iteration",
    )

    args = parser.parse_args()
    main(args)
