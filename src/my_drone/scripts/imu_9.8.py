#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu

G = 9.805

class ImuFixer:
    def __init__(self):
        self.pub = rospy.Publisher("/livox/imu_correct", Imu, queue_size=200)
        self.sub = rospy.Subscriber("/livox/imu", Imu, self.callback, queue_size=200)

    def callback(self, msg):
        out = Imu()
        out.header = msg.header

        # Livox IMU orientation 无效，保持原样即可
        out.orientation = msg.orientation
        out.orientation_covariance = msg.orientation_covariance

        # angular_velocity 看起来已经是 rad/s，先不改
        out.angular_velocity = msg.angular_velocity
        out.angular_velocity_covariance = msg.angular_velocity_covariance

        # 关键：加速度从 g 转成 m/s^2
        out.linear_acceleration.x = msg.linear_acceleration.x * G
        out.linear_acceleration.y = msg.linear_acceleration.y * G
        out.linear_acceleration.z = msg.linear_acceleration.z * G
        out.linear_acceleration_covariance = msg.linear_acceleration_covariance

        self.pub.publish(out)

if __name__ == "__main__":
    rospy.init_node("imu_fixer")
    ImuFixer()
    rospy.spin()