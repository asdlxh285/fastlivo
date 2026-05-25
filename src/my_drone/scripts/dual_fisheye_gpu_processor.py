#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import message_filters
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class RectifyCache:
    size: Tuple[int, int]
    new_camera_matrix: np.ndarray
    map1_cpu: np.ndarray
    map2_cpu: np.ndarray
    map1_gpu: Optional[object] = None
    map2_gpu: Optional[object] = None


class DualFisheyeGpuProcessor:
    """GPU accelerated preprocessor for two fisheye ROS image streams.

    ROS messages still enter and leave through host memory. The expensive per-pixel
    parts, color conversion and fisheye remap, use OpenCV CUDA when available. If
    OpenCV has no CUDA module or a CUDA call fails, the node falls back to CPU so
    the launch does not crash in the field.
    """

    def __init__(self):
        self.left_image_topic = rospy.get_param("~left_image_topic", "/camera/fisheye_left/image_raw")
        self.right_image_topic = rospy.get_param("~right_image_topic", "/camera/fisheye_right/image_raw")
        self.left_camera_info_topic = rospy.get_param("~left_camera_info_topic", "/camera/fisheye_left/camera_info")
        self.right_camera_info_topic = rospy.get_param("~right_camera_info_topic", "/camera/fisheye_right/camera_info")

        self.left_output_topic = rospy.get_param("~left_output_topic", "/camera_gpu/fisheye_left/image_rect")
        self.right_output_topic = rospy.get_param("~right_output_topic", "/camera_gpu/fisheye_right/image_rect")
        self.left_info_output_topic = rospy.get_param("~left_info_output_topic", "/camera_gpu/fisheye_left/camera_info")
        self.right_info_output_topic = rospy.get_param("~right_info_output_topic", "/camera_gpu/fisheye_right/camera_info")

        self.enable_rectify = bool(rospy.get_param("~enable_rectify", True))
        self.output_encoding = rospy.get_param("~output_encoding", "mono8").lower()
        self.interpolation = int(rospy.get_param("~interpolation", cv2.INTER_LINEAR))
        self.border_mode = int(rospy.get_param("~border_mode", cv2.BORDER_CONSTANT))
        self.balance = float(rospy.get_param("~balance", 0.0))
        self.fov_scale = float(rospy.get_param("~fov_scale", 1.0))
        self.use_gpu = bool(rospy.get_param("~use_gpu", True))
        self.gpu_device_id = int(rospy.get_param("~gpu_device_id", 0))
        self.approx_sync_slop = float(rospy.get_param("~approx_sync_slop", 0.02))
        self.queue_size = int(rospy.get_param("~queue_size", 8))
        self.single_stream_mode = bool(rospy.get_param("~single_stream_mode", False))

        if self.output_encoding not in ("mono8", "bgr8", "passthrough"):
            rospy.logwarn("[dual_fisheye_gpu] unsupported output_encoding=%s, fallback to mono8", self.output_encoding)
            self.output_encoding = "mono8"

        self.cuda_enabled = self.init_cuda()
        self.lock = threading.Lock()
        self.left_info: Optional[CameraInfo] = None
        self.right_info: Optional[CameraInfo] = None
        self.left_cache: Optional[RectifyCache] = None
        self.right_cache: Optional[RectifyCache] = None

        self.left_pub = rospy.Publisher(self.left_output_topic, Image, queue_size=self.queue_size)
        self.right_pub = rospy.Publisher(self.right_output_topic, Image, queue_size=self.queue_size)
        self.left_info_pub = rospy.Publisher(self.left_info_output_topic, CameraInfo, queue_size=self.queue_size)
        self.right_info_pub = rospy.Publisher(self.right_info_output_topic, CameraInfo, queue_size=self.queue_size)

        rospy.Subscriber(self.left_camera_info_topic, CameraInfo, self.left_info_callback, queue_size=self.queue_size)
        rospy.Subscriber(self.right_camera_info_topic, CameraInfo, self.right_info_callback, queue_size=self.queue_size)

        if self.single_stream_mode:
            rospy.Subscriber(self.left_image_topic, Image, self.left_image_callback, queue_size=self.queue_size)
            rospy.Subscriber(self.right_image_topic, Image, self.right_image_callback, queue_size=self.queue_size)
        else:
            left_sub = message_filters.Subscriber(self.left_image_topic, Image)
            right_sub = message_filters.Subscriber(self.right_image_topic, Image)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [left_sub, right_sub], queue_size=self.queue_size, slop=self.approx_sync_slop
            )
            self.sync.registerCallback(self.stereo_callback)

        rospy.loginfo("[dual_fisheye_gpu] left:  %s -> %s", self.left_image_topic, self.left_output_topic)
        rospy.loginfo("[dual_fisheye_gpu] right: %s -> %s", self.right_image_topic, self.right_output_topic)
        rospy.loginfo("[dual_fisheye_gpu] CUDA path: %s", "enabled" if self.cuda_enabled else "disabled/fallback")

    def init_cuda(self) -> bool:
        if not self.use_gpu or not hasattr(cv2, "cuda"):
            return False
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count <= 0:
                rospy.logwarn("[dual_fisheye_gpu] OpenCV CUDA module found, but no CUDA device is available")
                return False
            device_id = max(0, min(self.gpu_device_id, count - 1))
            cv2.cuda.setDevice(device_id)
            rospy.loginfo("[dual_fisheye_gpu] using CUDA device %d/%d", device_id, count)
            return True
        except Exception as error:
            rospy.logwarn("[dual_fisheye_gpu] CUDA init failed, using CPU fallback: %s", str(error))
            return False

    def left_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.left_info = msg
            self.left_cache = None

    def right_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.right_info = msg
            self.right_cache = None

    def left_image_callback(self, msg: Image):
        with self.lock:
            info = copy.deepcopy(self.left_info)
        self.process_and_publish(msg, info, "left")

    def right_image_callback(self, msg: Image):
        with self.lock:
            info = copy.deepcopy(self.right_info)
        self.process_and_publish(msg, info, "right")

    def stereo_callback(self, left_msg: Image, right_msg: Image):
        with self.lock:
            left_info = copy.deepcopy(self.left_info)
            right_info = copy.deepcopy(self.right_info)
        self.process_and_publish(left_msg, left_info, "left")
        self.process_and_publish(right_msg, right_info, "right")

    def validate_msg(self, msg: Image) -> bool:
        if msg.width <= 0 or msg.height <= 0 or msg.step <= 0:
            rospy.logwarn_throttle(2.0, "[dual_fisheye_gpu] invalid image shape %dx%d step=%d", msg.width, msg.height, msg.step)
            return False
        expected = int(msg.height) * int(msg.step)
        if len(msg.data) < expected:
            rospy.logwarn_throttle(
                2.0, "[dual_fisheye_gpu] truncated image data: got %d bytes, expected at least %d", len(msg.data), expected
            )
            return False
        return True

    def process_and_publish(self, msg: Image, info: Optional[CameraInfo], side: str):
        if not self.validate_msg(msg):
            return
        try:
            frame, current_encoding = self.ros_image_to_numpy(msg)
            if frame is None:
                return

            frame = self.convert_encoding(frame, current_encoding)

            cache = None
            if self.enable_rectify and info is not None:
                cache = self.get_rectify_cache(info, msg.width, msg.height, side)
                if cache is not None:
                    frame = self.remap(frame, cache)

            out_encoding = self.encoding_for_frame(frame)
            out_msg = self.numpy_to_ros_image(frame, msg.header, out_encoding)
            info_msg = self.rectified_info(info, msg.header, cache) if info is not None else None

            if side == "left":
                self.left_pub.publish(out_msg)
                if info_msg is not None:
                    self.left_info_pub.publish(info_msg)
            else:
                self.right_pub.publish(out_msg)
                if info_msg is not None:
                    self.right_info_pub.publish(info_msg)
        except Exception as error:
            rospy.logerr_throttle(2.0, "[dual_fisheye_gpu] %s image processing failed: %s", side, str(error))

    def ros_image_to_numpy(self, msg: Image):
        encoding = msg.encoding.lower()
        height = msg.height
        width = msg.width
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding in ("bgr8", "rgb8", "8uc3"):
            frame = data.reshape((height, msg.step))[:, :width * 3].reshape((height, width, 3))
            return frame.copy(), "bgr8" if encoding == "8uc3" else encoding
        if encoding in ("mono8", "8uc1"):
            frame = data.reshape((height, msg.step))[:, :width]
            return frame.copy(), "mono8"
        if encoding in ("bgra8", "rgba8"):
            frame = data.reshape((height, msg.step))[:, :width * 4].reshape((height, width, 4))
            return frame.copy(), encoding

        rospy.logwarn_throttle(2.0, "[dual_fisheye_gpu] unsupported image encoding: %s", msg.encoding)
        return None, encoding

    def convert_encoding(self, frame: np.ndarray, current_encoding: str) -> np.ndarray:
        if self.output_encoding == "passthrough":
            return frame

        if self.output_encoding == "mono8":
            if current_encoding == "mono8":
                return frame
            code = {
                "bgr8": cv2.COLOR_BGR2GRAY,
                "rgb8": cv2.COLOR_RGB2GRAY,
                "bgra8": cv2.COLOR_BGRA2GRAY,
                "rgba8": cv2.COLOR_RGBA2GRAY,
            }.get(current_encoding)
            return self.cuda_cvt_color(frame, code) if code is not None else frame

        if current_encoding == "bgr8":
            return frame
        code = {
            "rgb8": cv2.COLOR_RGB2BGR,
            "mono8": cv2.COLOR_GRAY2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR,
            "rgba8": cv2.COLOR_RGBA2BGR,
        }.get(current_encoding)
        return self.cuda_cvt_color(frame, code) if code is not None else frame

    def cuda_cvt_color(self, frame: np.ndarray, code: int) -> np.ndarray:
        if not self.cuda_enabled:
            return cv2.cvtColor(frame, code)
        try:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame)
            converted = cv2.cuda.cvtColor(gpu, code)
            return converted.download()
        except Exception as error:
            self.cuda_enabled = False
            rospy.logwarn("[dual_fisheye_gpu] CUDA cvtColor failed, switching to CPU: %s", str(error))
            return cv2.cvtColor(frame, code)

    def get_rectify_cache(self, info: CameraInfo, width: int, height: int, side: str) -> Optional[RectifyCache]:
        with self.lock:
            cache = self.left_cache if side == "left" else self.right_cache
            if cache is not None and cache.size == (width, height):
                return cache

            if len(info.K) < 9 or len(info.D) < 4:
                rospy.logwarn_throttle(2.0, "[dual_fisheye_gpu] %s CameraInfo has no fisheye intrinsics; skip rectify", side)
                return None

            k = np.array(info.K, dtype=np.float64).reshape(3, 3)
            d = np.array(info.D[:4], dtype=np.float64).reshape(4, 1)
            size = (int(width), int(height))
            r = np.eye(3, dtype=np.float64)
            try:
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    k, d, size, r, balance=self.balance, fov_scale=self.fov_scale
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, r, new_k, size, cv2.CV_32FC1)
            except Exception as error:
                rospy.logwarn_throttle(2.0, "[dual_fisheye_gpu] %s fisheye map build failed: %s", side, str(error))
                return None

            map1_gpu = None
            map2_gpu = None
            if self.cuda_enabled:
                try:
                    map1_gpu = cv2.cuda_GpuMat()
                    map2_gpu = cv2.cuda_GpuMat()
                    map1_gpu.upload(map1)
                    map2_gpu.upload(map2)
                except Exception as error:
                    self.cuda_enabled = False
                    rospy.logwarn("[dual_fisheye_gpu] CUDA map upload failed, switching to CPU: %s", str(error))
                    map1_gpu = None
                    map2_gpu = None

            cache = RectifyCache(size, new_k, map1, map2, map1_gpu, map2_gpu)
            if side == "left":
                self.left_cache = cache
            else:
                self.right_cache = cache
            return cache

    def remap(self, frame: np.ndarray, cache: RectifyCache) -> np.ndarray:
        if self.cuda_enabled and cache.map1_gpu is not None and cache.map2_gpu is not None:
            try:
                gpu = cv2.cuda_GpuMat()
                gpu.upload(frame)
                rectified = cv2.cuda.remap(gpu, cache.map1_gpu, cache.map2_gpu, self.interpolation, borderMode=self.border_mode)
                return rectified.download()
            except Exception as error:
                self.cuda_enabled = False
                rospy.logwarn("[dual_fisheye_gpu] CUDA remap failed, switching to CPU: %s", str(error))
        return cv2.remap(frame, cache.map1_cpu, cache.map2_cpu, self.interpolation, borderMode=self.border_mode)

    def encoding_for_frame(self, frame: np.ndarray) -> str:
        if frame.ndim == 2:
            return "mono8"
        if frame.ndim == 3 and frame.shape[2] == 3:
            return "bgr8"
        if frame.ndim == 3 and frame.shape[2] == 4:
            return "bgra8"
        return "passthrough"

    def numpy_to_ros_image(self, frame: np.ndarray, header, encoding: str) -> Image:
        contiguous = np.ascontiguousarray(frame)
        out = Image()
        out.header = header
        out.height = int(contiguous.shape[0])
        out.width = int(contiguous.shape[1])
        out.encoding = encoding
        out.is_bigendian = 0
        out.step = int(contiguous.strides[0])
        out.data = contiguous.tobytes()
        return out

    def rectified_info(self, info: CameraInfo, header, cache: Optional[RectifyCache]) -> CameraInfo:
        out = copy.deepcopy(info)
        out.header = header
        if cache is not None:
            k = cache.new_camera_matrix
            out.width = cache.size[0]
            out.height = cache.size[1]
            out.distortion_model = "plumb_bob"
            out.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            out.K = [float(k[0, 0]), float(k[0, 1]), float(k[0, 2]),
                     float(k[1, 0]), float(k[1, 1]), float(k[1, 2]),
                     float(k[2, 0]), float(k[2, 1]), float(k[2, 2])]
            out.R = [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]
            out.P = [float(k[0, 0]), float(k[0, 1]), float(k[0, 2]), 0.0,
                     float(k[1, 0]), float(k[1, 1]), float(k[1, 2]), 0.0,
                     float(k[2, 0]), float(k[2, 1]), float(k[2, 2]), 0.0]
        return out


if __name__ == "__main__":
    rospy.init_node("dual_fisheye_gpu_processor")
    DualFisheyeGpuProcessor()
    rospy.spin()
