#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file provides the ROS2Camera class for capturing frames over topics published by ROS 2 drivers for cameras.
"""

import logging
import threading
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import rclpy
import rclpy.node
import rclpy.subscription
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageMsg

from lerobot.common.cameras.camera import Camera
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robot_devices.cameras.configs import ColorMode, ROS2CameraConfig
from lerobot.common.utils.utils import capture_timestamp_utc

logger = logging.getLogger(__name__)


class ROS2Camera(Camera):
    """
    Manages camera interactions using ROS 2, adhering to the standard Camera interface.
    This class subscribes to a ROS 2 topic to receive and process image frames.

    Basic usage:
    ```python
    from lerobot.common.robot_devices.cameras.ros2 import ROS2Camera
    from lerobot.common.robot_devices.cameras.configs import ROS2CameraConfig

    config = ROS2CameraConfig(topic="/camera/image_raw")
    camera = ROS2Camera(config)
    camera.connect()
    image = camera.read()
    camera.disconnect()
    ```
    """

    # Static variables for managing the global rclpy context and instances
    _rclpy_initialized: bool = False
    _rclpy_node: rclpy.node.Node | None = None
    _rclpy_spin_thread: threading.Thread | None = None
    _rclpy_stop_event: threading.Event | None = None
    _cv_bridge: CvBridge | None = None
    _active_instances = 0
    _class_lock = threading.Lock()

    def __init__(self, config: ROS2CameraConfig):
        """Initializes the ROS2Camera instance."""
        super().__init__(config)
        self.config = config

        with ROS2Camera._class_lock:
            if ROS2Camera._active_instances == 0:
                self._initialize_rclpy()
            ROS2Camera._active_instances += 1

        self.subscription: rclpy.subscription.Subscription | None = None
        self.latest_msg: ImageMsg | None = None
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    @classmethod
    def _initialize_rclpy(cls):
        """Initializes the shared rclpy node and starts its spin thread."""
        logger.debug("Initializing rclpy for the first ROS2Camera instance.")
        if not cls._rclpy_initialized:
            rclpy.init()
            cls._rclpy_initialized = True
        cls._rclpy_node = rclpy.create_node("lerobot_camera_node")
        cls._rclpy_stop_event = threading.Event()
        cls._rclpy_spin_thread = threading.Thread(target=cls._spin_node, daemon=True)
        cls._rclpy_spin_thread.start()
        cls._cv_bridge = CvBridge()

    @classmethod
    def _shutdown_rclpy(cls):
        """Stops the spin thread and shuts down rclpy if no instances are left."""
        logger.debug("Shutting down rclpy as the last ROS2Camera instance is being destroyed.")
        if cls._rclpy_stop_event:
            cls._rclpy_stop_event.set()
        if cls._rclpy_spin_thread and cls._rclpy_spin_thread.is_alive():
            cls._rclpy_spin_thread.join(timeout=1.0)
        if cls._rclpy_node:
            cls._rclpy_node.destroy_node()
        if cls._rclpy_initialized:
            rclpy.shutdown()
            cls._rclpy_initialized = False
        cls._rclpy_node = None
        cls._cv_bridge = None
        logger.info("ROS 2 node for cameras shut down complete.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.topic})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera subscription is active."""
        return self.subscription is not None

    @staticmethod
    def _spin_node():
        """Spins the ROS 2 node to process callbacks until the stop event is set."""
        while rclpy.ok() and not ROS2Camera._rclpy_stop_event.is_set():
            rclpy.spin_once(ROS2Camera._rclpy_node, timeout_sec=0.1)

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        """Detects available ROS 2 camera topics on the network."""
        temp_node = None
        if not ROS2Camera._rclpy_initialized:
            rclpy.init()
            temp_node = rclpy.create_node("temp_camera_finder")
            node = temp_node
        else:
            node = ROS2Camera._rclpy_node

        time.sleep(1.0)
        topic_names_and_types = node.get_topic_names_and_types()

        if temp_node:
            temp_node.destroy_node()
            rclpy.shutdown()

        camera_topics = []
        for topic_name, msg_types in topic_names_and_types:
            if "sensor_msgs/msg/Image" in msg_types:
                camera_topics.append({
                    "name": f"ROS2 Camera @ {topic_name}",
                    "type": "ROS2",
                    "id": topic_name,
                })
        return camera_topics

    def connect(self, warmup: bool = True) -> None:
        """Establishes connection to the camera topic."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.subscription = self._rclpy_node.create_subscription(
            ImageMsg, self.config.topic, self._sub_cb, 10
        )

        if warmup:
            logger.info(f"Warming up {self}... Waiting for the first message.")
            if not self.new_frame_event.wait(timeout=10.0):
                self.disconnect()
                raise TimeoutError(f"Timeout waiting for first message on topic '{self.config.topic}'")

        logger.info(f"{self} connected.")

    def _sub_cb(self, msg: ImageMsg):
        """Internal callback to handle incoming image messages."""
        with self.frame_lock:
            self.latest_msg = msg
        self.new_frame_event.set()

        if self.width is None or self.height is None:
            self.width, self.height = msg.width, msg.height

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """Captures and returns a single frame by waiting for the next available one."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Wait for a new frame to be delivered by the subscription callback
        if not self.new_frame_event.wait(timeout=1.0):  # 1-second timeout for a frame
            raise TimeoutError(f"Timed out waiting for a new frame from {self}.")

        with self.frame_lock:
            if self.latest_msg is None:
                raise RuntimeError(f"Internal error: Event set but no message available for {self}.")
            # Create a copy of the message to process
            msg_to_process = self.latest_msg
            self.new_frame_event.clear()

        return self._postprocess_image(msg_to_process, color_mode)

    def _postprocess_image(self, msg: ImageMsg, color_mode: ColorMode | None) -> np.ndarray:
        """Applies color conversion and rotation to a raw message frame."""
        # Use config's encoding. The color_mode arg from base class is ignored
        # as ROS publishers define their own encoding.
        image = self._cv_bridge.imgmsg_to_cv2(msg, self.config.encoding)

        if self.rotation is not None:
            image = cv2.rotate(image, self.rotation)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]

        return image

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """Asynchronously captures a frame, waiting for a new one if necessary."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms.")

        with self.frame_lock:
            if self.latest_msg is None:
                raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
            msg_to_process = self.latest_msg
            self.new_frame_event.clear()

        return self._postprocess_image(msg_to_process, None)

    def disconnect(self) -> None:
        """Disconnects from the camera topic and releases resources."""
        if not self.is_connected:
            # Avoid raising an error if already disconnected to allow idempotent calls.
            return

        self._rclpy_node.destroy_subscription(self.subscription)
        self.subscription = None
        logger.info(f"{self} disconnected.")

    def __del__(self):
        """Destructor to ensure disconnection and cleanup of ROS 2 context."""
        try:
            if self.is_connected:
                self.disconnect()
        finally:
            with ROS2Camera._class_lock:
                ROS2Camera._active_instances -= 1
                if ROS2Camera._active_instances == 0:
                    self._shutdown_rclpy()
