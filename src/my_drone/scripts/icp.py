#!/usr/bin/python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import copy
import _thread
import time
from collections import deque

import open3d as o3d
import rospy
import numpy as np
np.float = float

import ros_numpy
import sensor_msgs.point_cloud2 as pc2

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import tf
import tf.transformations


global_map = None
global_map_np = None
initialized = False

# 语义：
# T_map_to_odom:     map_frame -> odom_frame
# T_icp_map_to_odom: icp_map_frame -> odom_frame
T_map_to_odom = np.eye(4)
T_icp_map_to_odom = np.eye(4)

cur_odom = None
cur_scan = None

# 初始化阶段用于保存“map->odom”的初值
initial_pose = None

scan_queue = None
tf_broadcaster = None
warned_body_frame = False
last_tf_stamp_ns = {}
PUBLISH_BODY_TF = False


def normalize_angle_rad(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def yaw_from_matrix(transform):
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def inverse_se3(trans):
    trans_inverse = np.eye(4)
    trans_inverse[:3, :3] = trans[:3, :3].T
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def msg_to_array(pc_msg):
    if pc_msg is None:
        return np.empty((0, 3), dtype=np.float32)

    if len(pc_msg.fields) == 0 or pc_msg.point_step == 0 or pc_msg.width == 0:
        return np.empty((0, 3), dtype=np.float32)

    try:
        points = np.array(
            list(pc2.read_points(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True)),
            dtype=np.float32
        )
    except Exception as err:
        rospy.logwarn_throttle(2.0, 'Invalid PointCloud2 frame skipped: {}'.format(err))
        return np.empty((0, 3), dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)

    return points


def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except Exception:
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale),
        voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale,
        initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    return result_icp.transformation, result_icp.fitness


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def broadcast_tf_from_matrix(transform_matrix, stamp, parent_frame, child_frame):
    global last_tf_stamp_ns
    if tf_broadcaster is None:
        return
    edge = (parent_frame, child_frame)
    stamp_ns = stamp.to_nsec()
    last_ns = last_tf_stamp_ns.get(edge)
    if last_ns is not None and stamp_ns <= last_ns:
        return
    last_tf_stamp_ns[edge] = stamp_ns
    xyz = tf.transformations.translation_from_matrix(transform_matrix)
    quat = tf.transformations.quaternion_from_matrix(transform_matrix)
    tf_broadcaster.sendTransform(xyz, quat, stamp, child_frame, parent_frame)


def get_effective_odom_frame():
    if ODOM_FRAME:
        return ODOM_FRAME
    if cur_odom is not None and cur_odom.header.frame_id:
        return cur_odom.header.frame_id
    return 'odom'


def get_effective_body_frame():
    if BASE_FRAME:
        return BASE_FRAME
    if cur_odom is not None and cur_odom.child_frame_id:
        return cur_odom.child_frame_id
    return 'base_link'


def cb_initialpose(msg):
    global initial_pose, initialized, T_map_to_odom, T_icp_map_to_odom, cur_odom

    if msg.header.frame_id and msg.header.frame_id != MAP_FRAME:
        rospy.logwarn(
            'Initialpose frame is "%s", but map_frame is "%s". Set RViz Fixed Frame to "%s" before 2D Pose Estimate.',
            msg.header.frame_id, MAP_FRAME, MAP_FRAME
        )

    if cur_odom is None:
        rospy.logwarn(
            'Received /initialpose but no odom yet. Wait for odom first, then send initial pose again.'
        )
        return

    # RViz /initialpose 表示 “机器人(base/body)在 map_frame 下的位姿”
    T_map_to_body_init = pose_to_mat(msg)

    # 当前里程计表示 “odom -> body”
    T_odom_to_body = pose_to_mat(cur_odom)

    # 程序真正需要的是 map -> odom
    # T_map_to_body = T_map_to_odom * T_odom_to_body
    # => T_map_to_odom = T_map_to_body * inverse(T_odom_to_body)
    T_map_to_odom = np.matmul(T_map_to_body_init, inverse_se3(T_odom_to_body))
    T_icp_map_to_odom = T_map_to_odom.copy()
    initial_pose = T_map_to_odom.copy()

    initialized = False

    yaw_body_init = yaw_from_matrix(T_map_to_body_init)
    yaw_odom_body = yaw_from_matrix(T_odom_to_body)
    yaw_map_odom = yaw_from_matrix(T_map_to_odom)

    rospy.logwarn(
        'Received new initial pose. Reset relocalization. yaw(map->body)=%.2f deg, yaw(odom->body)=%.2f deg, computed yaw(map->odom)=%.2f deg',
        np.degrees(yaw_body_init),
        np.degrees(yaw_odom_body),
        np.degrees(yaw_map_odom),
    )


def crop_global_map_in_FOV(pose_estimation, odom_msg):
    global global_map_np

    T_odom_to_body = pose_to_mat(odom_msg)
    T_map_to_body = np.matmul(pose_estimation, T_odom_to_body)
    t_robot = T_map_to_body[:3, 3]

    mask = (
        (global_map_np[:, 0] > t_robot[0] - FOV_FAR) & (global_map_np[:, 0] < t_robot[0] + FOV_FAR) &
        (global_map_np[:, 1] > t_robot[1] - FOV_FAR) & (global_map_np[:, 1] < t_robot[1] + FOV_FAR) &
        (global_map_np[:, 2] > t_robot[2] - FOV_FAR) & (global_map_np[:, 2] < t_robot[2] + FOV_FAR)
    )
    local_map_np = global_map_np[mask]

    if len(local_map_np) > 0:
        if FOV > 3.14:
            dist_sq = (
                (local_map_np[:, 0] - t_robot[0]) ** 2 +
                (local_map_np[:, 1] - t_robot[1]) ** 2 +
                (local_map_np[:, 2] - t_robot[2]) ** 2
            )
            indices = np.where(dist_sq < FOV_FAR ** 2)
            local_map_np = local_map_np[indices]
        else:
            T_body_to_map = inverse_se3(T_map_to_body)
            local_map_in_body = np.matmul(T_body_to_map[:3, :3], local_map_np.T).T + T_body_to_map[:3, 3]
            dist_sq = (
                local_map_in_body[:, 0] ** 2 +
                local_map_in_body[:, 1] ** 2 +
                local_map_in_body[:, 2] ** 2
            )
            indices = np.where(
                (local_map_in_body[:, 0] > 0) &
                (dist_sq < FOV_FAR ** 2) &
                (np.abs(np.arctan2(local_map_in_body[:, 1], local_map_in_body[:, 0])) < FOV / 2.0)
            )
            local_map_np = local_map_np[indices]

    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(local_map_np)

    if len(local_map_np) > 0:
        header = odom_msg.header
        header.frame_id = MAP_FRAME
        publish_point_cloud(pub_submap, header, local_map_np[::10])

    return global_map_in_FOV


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom, T_icp_map_to_odom, initialized

    if cur_odom is None:
        rospy.logwarn_throttle(1.0, 'Skip ICP: odom not ready')
        return False
    if cur_scan is None:
        rospy.logwarn_throttle(1.0, 'Skip ICP: scan not ready')
        return False

    rospy.loginfo('Global localization by scan-to-map matching......')

    scan_tobe_mapped = copy.deepcopy(cur_scan)
    tic = time.time()

    scan_points = len(np.asarray(scan_tobe_mapped.points))
    if scan_points < MIN_SCAN_POINTS:
        rospy.logwarn_throttle(1.0, 'Skip ICP: scan points too few ({})'.format(scan_points))
        return False

    global_map_in_FOV = crop_global_map_in_FOV(pose_estimation, cur_odom)
    submap_points = len(np.asarray(global_map_in_FOV.points))

    if submap_points < MIN_SUBMAP_POINTS and GLOBAL_FALLBACK:
        rospy.logwarn_throttle(
            1.0,
            'Local submap too small ({}) -> fallback to global map ({})'.format(
                submap_points, len(global_map_np)
            )
        )
        global_map_in_FOV = global_map
        submap_points = len(np.asarray(global_map_in_FOV.points))

    if submap_points < MIN_SUBMAP_POINTS:
        rospy.logwarn_throttle(1.0, 'Skip ICP: submap points too few ({})'.format(submap_points))
        return False

    transformation, _ = registration_at_scale(
        scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5
    )
    transformation, fitness = registration_at_scale(
        scan_tobe_mapped, global_map_in_FOV, initial=transformation, scale=1
    )

    toc = time.time()
    rospy.loginfo('Time elapsed: {:.4f} s'.format(toc - tic))
    rospy.loginfo('fitness: {:.6f}, threshold: {:.6f}'.format(fitness, LOCALIZATION_TH))
    rospy.loginfo('')

    # Open3D fitness 越大通常越好，所以这里维持“fitness > threshold 才算成功”
    if fitness > LOCALIZATION_TH:
        if not initialized:
            yaw_init = yaw_from_matrix(pose_estimation)
            yaw_est = yaw_from_matrix(transformation)
            yaw_err_deg = abs(np.degrees(normalize_angle_rad(yaw_est - yaw_init)))
            if yaw_err_deg > MAX_INIT_YAW_ERR_DEG:
                rospy.logwarn(
                    'Reject init match: yaw diff %.2f deg > %.2f deg (fitness=%.4f)',
                    yaw_err_deg, MAX_INIT_YAW_ERR_DEG, fitness
                )
                return False

        T_map_to_odom = transformation
        T_icp_map_to_odom = transformation.copy()

        xyz = tf.transformations.translation_from_matrix(T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)

        map_to_odom = Odometry()
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = MAP_FRAME
        map_to_odom.child_frame_id = get_effective_odom_frame()
        pub_map_to_odom.publish(map_to_odom)

        icp_map_to_odom = Odometry()
        icp_map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        icp_map_to_odom.header.stamp = cur_odom.header.stamp
        icp_map_to_odom.header.frame_id = ICP_MAP_FRAME
        icp_map_to_odom.child_frame_id = get_effective_odom_frame()
        pub_icp_map_to_odom.publish(icp_map_to_odom)

        if PUBLISH_ICP_TF:
            broadcast_tf_from_matrix(
                T_icp_map_to_odom,
                cur_odom.header.stamp,
                ICP_MAP_FRAME,
                get_effective_odom_frame()
            )

        return True
    else:
        rospy.logwarn('Not match!!!! fitness score: {}'.format(fitness))
        return False


def initialize_global_map(pc_msg):
    global global_map, global_map_np
    rospy.logwarn('pre-processing global map ......')

    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)

    global_map_np = np.asarray(global_map.points)
    rospy.loginfo('Global map received. Map size: {} points'.format(len(global_map_np)))


def cb_save_cur_odom(odom_msg):
    global cur_odom, T_map_to_odom, T_icp_map_to_odom, initialized
    cur_odom = odom_msg

    # Keep the map frame visible in RViz before ICP initialization succeeds.
    # Before /initialpose this is identity; after /initialpose it becomes the
    # user's coarse guess; after ICP it is the refined transform.
    if PUBLISH_ICP_TF:
        odom_frame = get_effective_odom_frame()
        broadcast_tf_from_matrix(T_icp_map_to_odom, odom_msg.header.stamp, ICP_MAP_FRAME, odom_frame)

    if initialized:
        T_odom_to_body = pose_to_mat(odom_msg)
        T_map_to_body = np.matmul(T_map_to_odom, T_odom_to_body)

        odom_in_map = Odometry()
        odom_in_map.header.stamp = odom_msg.header.stamp
        odom_in_map.header.frame_id = MAP_FRAME
        odom_in_map.child_frame_id = get_effective_body_frame()

        xyz = tf.transformations.translation_from_matrix(T_map_to_body)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_body)
        odom_in_map.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        odom_in_map.twist = odom_msg.twist
        pub_odom_in_map.publish(odom_in_map)

        T_icp_map_to_body = np.matmul(T_icp_map_to_odom, T_odom_to_body)
        body_frame = get_effective_body_frame()

        odom_in_icp_map = Odometry()
        odom_in_icp_map.header.stamp = odom_msg.header.stamp
        odom_in_icp_map.header.frame_id = ICP_MAP_FRAME
        odom_in_icp_map.child_frame_id = body_frame

        xyz_icp = tf.transformations.translation_from_matrix(T_icp_map_to_body)
        quat_icp = tf.transformations.quaternion_from_matrix(T_icp_map_to_body)
        odom_in_icp_map.pose.pose = Pose(Point(*xyz_icp), Quaternion(*quat_icp))
        odom_in_icp_map.twist = odom_msg.twist
        pub_odom_in_icp_map.publish(odom_in_icp_map)

        if PUBLISH_ICP_TF and PUBLISH_BODY_TF and body_frame != odom_frame:
            broadcast_tf_from_matrix(T_icp_map_to_body, odom_msg.header.stamp, ICP_MAP_FRAME, body_frame)


def cb_save_cur_scan(pc_msg):
    global cur_scan, T_map_to_odom, initialized, scan_queue, warned_body_frame

    if (not warned_body_frame) and ('body' in pc_msg.header.frame_id.lower()):
        rospy.logwarn(
            'scan_topic appears to be body-frame (%s). This script expects odom/world-aligned scan (e.g. /cloud_registered).',
            pc_msg.header.frame_id
        )
        warned_body_frame = True

    pub_pc_in_map.publish(pc_msg)

    pc = msg_to_array(pc_msg)
    if pc.shape[0] == 0:
        rospy.logwarn_throttle(
            2.0,
            'Skip scan update: incoming scan on %s has 0 valid XYZ points',
            SCAN_TOPIC
        )
        return

    scan_queue.append(pc)
    local_map_points = np.vstack(scan_queue)

    local_map_pcd = o3d.geometry.PointCloud()
    local_map_pcd.points = o3d.utility.Vector3dVector(local_map_points)

    cur_scan = voxel_down_sample(local_map_pcd, SCAN_VOXEL_SIZE)

    if not initialized:
        rospy.logwarn_throttle(
            2.0,
            'Not publishing /realtime_scan_in_map yet: localization is not initialized. '
            'Wait for /initialpose and a successful ICP match first.'
        )
        return

    pc_local_ds = np.asarray(cur_scan.points)
    if len(pc_local_ds) == 0:
        rospy.logwarn_throttle(
            2.0,
            'Skip /realtime_scan_in_map: downsampled local scan is empty'
        )
        return

    pc_homo = np.column_stack([pc_local_ds, np.ones(len(pc_local_ds))])
    pc_in_map = np.matmul(T_map_to_odom, pc_homo.T).T[:, :3]

    header = copy.deepcopy(pc_msg.header)
    header.frame_id = MAP_FRAME
    publish_point_cloud(pub_realtime_scan, header, pc_in_map)


def thread_localization():
    global T_map_to_odom
    while not rospy.is_shutdown():
        rospy.sleep(1.0 / FREQ_LOCALIZATION)
        global_localization(T_map_to_odom)


if __name__ == '__main__':
    rospy.init_node('fast_lio_localization')

    MAP_VOXEL_SIZE = rospy.get_param('~map_voxel_size', 0.2)
    SCAN_VOXEL_SIZE = rospy.get_param('~scan_voxel_size', 0.1)

    FREQ_LOCALIZATION = rospy.get_param('~FREQ_LOCALIZATION', 0.5)
    LOCALIZATION_TH = rospy.get_param('~LOCALIZATION_TH', 0.95)
    MAX_INIT_YAW_ERR_DEG = rospy.get_param('~max_init_yaw_err_deg', 35.0)

    PUBLISH_ICP_TF = rospy.get_param('~publish_icp_tf', True)
    PUBLISH_BODY_TF = rospy.get_param('~publish_body_tf', False)

    ICP_MAP_FRAME = rospy.get_param('~icp_map_frame', 'icp_map')
    MAP_FRAME = rospy.get_param('~map_frame', 'map')
    ODOM_FRAME = rospy.get_param('~odom_frame', '')
    BASE_FRAME = rospy.get_param('~base_frame', '')

    FOV = rospy.get_param('~fov', 6.28)
    FOV_FAR = rospy.get_param('~fov_far', 40.0)
    MIN_SCAN_POINTS = rospy.get_param('~min_scan_points', 80)
    MIN_SUBMAP_POINTS = rospy.get_param('~min_submap_points', 120)
    GLOBAL_FALLBACK = rospy.get_param('~global_fallback', True)

    LOCAL_MAP_FRAMES = rospy.get_param('~LOCAL_MAP_FRAMES', 5)
    ODOM_TOPIC = rospy.get_param('~odom_topic', '/aft_mapped_to_init')
    SCAN_TOPIC = rospy.get_param('~scan_topic', '/cloud_freedom')
    MAP_TOPIC = rospy.get_param('~map_topic', '/map')

    scan_queue = deque(maxlen=LOCAL_MAP_FRAMES)

    rospy.loginfo(
        'Localization Node Inited. Local map frames: {}, odom topic: {}, scan topic: {}, map topic: {}, '
        'map_frame: {}, odom_frame: {}, base_frame: {}, icp_map_frame: {}, fov_far: {}, global_fallback: {}, '
        'th: {}, map_voxel: {}, scan_voxel: {}, max_init_yaw_err_deg: {}'.format(
            LOCAL_MAP_FRAMES, ODOM_TOPIC, SCAN_TOPIC, MAP_TOPIC,
            MAP_FRAME, ODOM_FRAME, BASE_FRAME, ICP_MAP_FRAME,
            FOV_FAR, GLOBAL_FALLBACK, LOCALIZATION_TH,
            MAP_VOXEL_SIZE, SCAN_VOXEL_SIZE, MAX_INIT_YAW_ERR_DEG
        )
    )

    tf_broadcaster = tf.TransformBroadcaster()

    pub_pc_in_map = rospy.Publisher('/cur_scan_in_odom', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)

    pub_odom_in_map = rospy.Publisher('/odom_in_map', Odometry, queue_size=1)
    pub_icp_map_to_odom = rospy.Publisher('/icp_map_to_odom', Odometry, queue_size=1)
    pub_odom_in_icp_map = rospy.Publisher('/odom_in_icp_map', Odometry, queue_size=1)
    pub_realtime_scan = rospy.Publisher('/realtime_scan_in_map', PointCloud2, queue_size=1)

    rospy.Subscriber(SCAN_TOPIC, PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber(ODOM_TOPIC, Odometry, cb_save_cur_odom, queue_size=1)
    rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, cb_initialpose, queue_size=1)

    rospy.logwarn('Waiting for global map......')
    initialize_global_map(rospy.wait_for_message(MAP_TOPIC, PointCloud2))

    rospy.logwarn('Waiting for odom......')
    rospy.wait_for_message(ODOM_TOPIC, Odometry)
    while cur_odom is None and not rospy.is_shutdown():
        rospy.sleep(0.05)

    rospy.logwarn('Waiting for initial pose....')
    pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
    cb_initialpose(pose_msg)
    rospy.loginfo('Initial pose received successfully.')

    rate = rospy.Rate(1.0)
    max_retries = 10
    retry_count = 0

    while not initialized and not rospy.is_shutdown():
        if retry_count >= max_retries:
            rospy.logerr(
                'Initialization failed after {} attempts! Please provide a new initial pose in RViz.'.format(max_retries)
            )
            pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
            cb_initialpose(pose_msg)
            rospy.loginfo('New initial pose received successfully.')
            retry_count = 0
            continue

        if cur_scan is None or len(scan_queue) == 0:
            rospy.logwarn('Waiting for the latest scan...')
            rate.sleep()
            continue

        if cur_odom is None:
            rospy.logwarn('Waiting for odom...')
            rate.sleep()
            continue

        retry_count += 1
        rospy.loginfo('Attempt {}/{} for global localization using the latest scan...'.format(
            retry_count, max_retries
        ))

        initialized = global_localization(initial_pose)

        if not initialized:
            rospy.logwarn('Localization failed (fitness too low or yaw rejected). Retrying in 1 second...')
            rate.sleep()

    rospy.loginfo('##################################')
    if initialized:
        rospy.loginfo('Global localization initialized successfully!')
    else:
        rospy.loginfo('Global localization initialized failed!')
    rospy.loginfo('##################################')

    _thread.start_new_thread(thread_localization, ())
    rospy.spin()
