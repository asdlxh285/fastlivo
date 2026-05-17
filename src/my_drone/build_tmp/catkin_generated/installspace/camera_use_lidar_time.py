#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

        self.latest_lidar_stamp = None
        self.latest_lidar_seq = 0

        self.image_pub = rospy.Publisher(self.image_out_topic, Image, queue_size=50)
        self.camera_info_pub = rospy.Publisher(self.camera_info_out_topic, CameraInfo, queue_size=50)

        rospy.Subscriber(self.lidar_topic, CustomMsg, self.lidar_callback, queue_size=200)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=200)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=200)

        rospy.loginfo("[camera_use_lidar_time] lidar_topic: %s", self.lidar_topic)
        rospy.loginfo("[camera_use_lidar_time] image_topic: %s -> %s", self.image_topic, self.image_out_topic)
        rospy.loginfo("[camera_use_lidar_time] camera_info_topic: %s -> %s", self.camera_info_topic, self.camera_info_out_topic)

    def lidar_callback(self, msg):
        self.latest_lidar_stamp = msg.header.stamp
        self.latest_lidar_seq = msg.header.seq

    def image_callback(self, msg):
        if self.latest_lidar_stamp is None:
            return

        out = Image()
        out.header = msg.header
        out.header.stamp = self.latest_lidar_stamp
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

        out = CameraInfo()
        out.header = msg.header
        out.header.stamp = self.latest_lidar_stamp
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
    rospy.init_node("camera_use_lidar_time")
    CameraUseLidarTime()
    rospy.spin()
