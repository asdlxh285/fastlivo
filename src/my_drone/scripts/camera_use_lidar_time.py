#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import fcntl
import rospy
from sensor_msgs.msg import Image, CameraInfo
from livox_ros_driver2.msg import CustomMsg


class CameraUseLidarTime:
    def __init__(self):
        self.lidar_topic = rospy.get_param("~lidar_topic", "/livox/lidar")
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.image_out_topic = rospy.get_param("~image_out_topic", "/camera_sync/color/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
        self.camera_info_out_topic = rospy.get_param("~camera_info_out_topic", "/camera_sync/color/camera_info")
        self.use_lidar_seq = rospy.get_param("~use_lidar_seq", False)
        self.stamp_mode = rospy.get_param("~stamp_mode", "offset")
        self.offset_alpha = float(rospy.get_param("~offset_alpha", 0.05))
        self.max_offset_jump = float(rospy.get_param("~max_offset_jump", 0.5))

        self.latest_lidar_stamp = None
        self.latest_lidar_seq = 0
        self.clock_offset = None

        self.image_pub = rospy.Publisher(self.image_out_topic, Image, queue_size=50)
        self.camera_info_pub = rospy.Publisher(self.camera_info_out_topic, CameraInfo, queue_size=50)

        rospy.Subscriber(self.lidar_topic, CustomMsg, self.lidar_callback, queue_size=200)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=200)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=200)

        rospy.loginfo("[camera_use_lidar_time] lidar_topic: %s", self.lidar_topic)
        rospy.loginfo("[camera_use_lidar_time] image_topic: %s -> %s", self.image_topic, self.image_out_topic)
        rospy.loginfo("[camera_use_lidar_time] camera_info_topic: %s -> %s", self.camera_info_topic, self.camera_info_out_topic)
        rospy.loginfo("[camera_use_lidar_time] stamp_mode: %s", self.stamp_mode)

    def lidar_callback(self, msg):
        receive_time = rospy.Time.now()
        self.latest_lidar_stamp = msg.header.stamp
        self.latest_lidar_seq = msg.header.seq
        observed_offset = (msg.header.stamp - receive_time).to_sec()
        if self.clock_offset is None or abs(observed_offset - self.clock_offset) > self.max_offset_jump:
            self.clock_offset = observed_offset
        else:
            alpha = max(0.0, min(1.0, self.offset_alpha))
            self.clock_offset = (1.0 - alpha) * self.clock_offset + alpha * observed_offset

    def translated_stamp(self, stamp):
        if self.stamp_mode == "passthrough":
            return stamp
        if self.stamp_mode == "latest_lidar":
            return self.latest_lidar_stamp
        if self.clock_offset is None:
            return None
        return stamp + rospy.Duration.from_sec(self.clock_offset)

    def image_callback(self, msg):
        if self.latest_lidar_stamp is None:
            return

        stamp = self.translated_stamp(msg.header.stamp)
        if stamp is None:
            return

        out = Image()
        out.header = msg.header
        out.header.stamp = stamp
        if self.use_lidar_seq:
            out.header.seq = self.latest_lidar_seq
        out.height = msg.height
        out.width = msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = msg.step
        out.data = msg.data
        self.image_pub.publish(out)

    def camera_info_callback(self, msg):
        if self.latest_lidar_stamp is None:
            return

        stamp = self.translated_stamp(msg.header.stamp)
        if stamp is None:
            return

        out = CameraInfo()
        out.header = msg.header
        out.header.stamp = stamp
        if self.use_lidar_seq:
            out.header.seq = self.latest_lidar_seq
        out.height = msg.height
        out.width = msg.width
        out.distortion_model = msg.distortion_model
        out.D = list(msg.D)
        out.K = list(msg.K)
        out.R = list(msg.R)
        out.P = list(msg.P)
        out.binning_x = msg.binning_x
        out.binning_y = msg.binning_y
        out.roi = msg.roi
        self.camera_info_pub.publish(out)


if __name__ == "__main__":
    # Ensure only one instance runs to avoid duplicate stamping + extra CPU.
    lock_path = "/tmp/camera_use_lidar_time.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("[camera_use_lidar_time] Another instance is already running. Exiting.\n")
        sys.exit(0)

    rospy.init_node("camera_use_lidar_time")
    CameraUseLidarTime()
    rospy.spin()
