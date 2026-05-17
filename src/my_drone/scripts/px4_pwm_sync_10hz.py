#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import ActuatorControl

if __name__ == '__main__':
    rospy.init_node('px4_aux6_10hz_sync', anonymous=True)
    pub = rospy.Publisher('/mavros/actuator_control', ActuatorControl, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz 方波频率
    msg = ActuatorControl()
    msg.group_mix = 0
    # 对应 RC AUX 1 通道
    msg.controls = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    while not rospy.is_shutdown():
        # 10Hz 频率切换高低电平
        msg.controls[0] = 1.0 if msg.controls[0] == -1.0 else -1.0
        pub.publish(msg)
        rate.sleep()
