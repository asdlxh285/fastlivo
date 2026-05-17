#!/usr/bin/python3

import os
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image


class CameraCaptureWithInfo:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.save_dir = os.path.expanduser(rospy.get_param("~save_dir", "~/a"))
        self.window_name = rospy.get_param("~window_name", "camera_preview")
        self.require_gui = rospy.get_param("~require_gui", True)

        os.makedirs(self.save_dir, exist_ok=True)

        self.lock = threading.Lock()

        self.last_image = None
        self.last_info = None

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=10)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.info_callback, queue_size=10)

        self.gui_enabled = self.init_gui()
        rospy.loginfo("[camera_capture_with_info] image_topic: %s", self.image_topic)
        rospy.loginfo("[camera_capture_with_info] camera_info_topic: %s", self.camera_info_topic)
        rospy.loginfo("[camera_capture_with_info] save_dir: %s", self.save_dir)

    def init_gui(self):
        display = os.environ.get("DISPLAY", "")
        if not display and os.path.exists("/tmp/.X11-unix/X0"):
            os.environ["DISPLAY"] = ":0"
            display = ":0"
            rospy.logwarn("[camera_capture_with_info] DISPLAY not set, fallback to :0")

        if not display:
            message = "[camera_capture_with_info] no DISPLAY found, cannot open OpenCV window"
            if self.require_gui:
                rospy.logfatal(message)
                rospy.logfatal("[camera_capture_with_info] run in desktop terminal or set DISPLAY, e.g. export DISPLAY=:0")
                raise RuntimeError(message)
            rospy.logwarn(message + ", continue without GUI")
            return False

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            return True
        except Exception as error:
            message = f"[camera_capture_with_info] OpenCV window init failed: {error}"
            if self.require_gui:
                rospy.logfatal(message)
                raise
            rospy.logwarn(message + ", continue without GUI")
            return False

    def image_callback(self, msg):
        frame = self.ros_image_to_bgr(msg)
        if frame is None:
            return
        with self.lock:
            self.last_image = frame

    def ros_image_to_bgr(self, msg):
        encoding = msg.encoding.lower()
        height = msg.height
        width = msg.width

        try:
            if encoding in ("bgr8", "8uc3"):
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((height, msg.step))[:, :width * 3].reshape((height, width, 3))
                return frame.copy()

            if encoding == "rgb8":
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((height, msg.step))[:, :width * 3].reshape((height, width, 3))
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if encoding in ("mono8", "8uc1"):
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((height, msg.step))[:, :width]
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if encoding == "bgra8":
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((height, msg.step))[:, :width * 4].reshape((height, width, 4))
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if encoding == "rgba8":
                data = np.frombuffer(msg.data, dtype=np.uint8)
                frame = data.reshape((height, msg.step))[:, :width * 4].reshape((height, width, 4))
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            rospy.logwarn_throttle(2.0, "[camera_capture_with_info] unsupported encoding: %s", msg.encoding)
            return None
        except Exception as error:
            rospy.logerr_throttle(2.0, "[camera_capture_with_info] decode image failed: %s", str(error))
            return None

    def info_callback(self, msg):
        with self.lock:
            self.last_info = msg

    def build_overlay(self):
        if self.last_info is None:
            return ["camera_info: waiting..."]

        info = self.last_info
        fx = info.K[0]
        fy = info.K[4]
        cx = info.K[2]
        cy = info.K[5]
        return [
            f"size: {info.width}x{info.height}",
            f"fx: {fx:.3f} fy: {fy:.3f}",
            f"cx: {cx:.3f} cy: {cy:.3f}",
            f"model: {info.distortion_model}",
            "SPACE: save  |  Q/ESC: quit",
        ]

    def save_current(self):
        with self.lock:
            if self.last_image is None:
                rospy.logwarn("[camera_capture_with_info] no image yet")
                return
            image_to_save = self.last_image.copy()

        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
        path = os.path.join(self.save_dir, filename)
        ok = cv2.imwrite(path, image_to_save)
        if ok:
            rospy.loginfo("[camera_capture_with_info] saved: %s", path)
        else:
            rospy.logerr("[camera_capture_with_info] failed to save: %s", path)

    def run(self):
        rate = rospy.Rate(60)
        while not rospy.is_shutdown():
            with self.lock:
                frame = None if self.last_image is None else self.last_image.copy()

            if frame is not None and self.gui_enabled:
                lines = self.build_overlay()
                y = 30
                for line in lines:
                    cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    y += 30
                cv2.imshow(self.window_name, frame)

            if self.gui_enabled:
                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    self.save_current()
                elif key == ord('q') or key == 27:
                    break

            rate.sleep()

        if self.gui_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("camera_capture_with_info")
    CameraCaptureWithInfo().run()
