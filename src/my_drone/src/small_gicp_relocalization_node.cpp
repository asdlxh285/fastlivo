#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <small_gicp/pcl/pcl_point.hpp>
#include <small_gicp/pcl/pcl_point_traits.hpp>
#include <small_gicp/pcl/pcl_registration.hpp>

#include <Eigen/Geometry>

#include <mutex>
#include <string>

class SmallGicpRelocalizationNode {
public:
  SmallGicpRelocalizationNode() : private_nh_("~") {
    private_nh_.param<std::string>("map_pcd_path", map_pcd_path_, "/home/uav/catkin_ws/src/FAST-LIVO2/Log/pcd/all_downsampled_points.pcd");
    private_nh_.param<std::string>("input_cloud_topic", input_cloud_topic_, "/cloud_registered");
    private_nh_.param<std::string>("output_odom_topic", output_odom_topic_, "/small_gicp/relocalization_odom");
    private_nh_.param<std::string>("map_frame", map_frame_, "map");
    private_nh_.param<std::string>("base_frame", base_frame_, "base_link");
    private_nh_.param<std::string>("tf_child_frame", tf_child_frame_, "small_gicp_base_link");
    private_nh_.param<int>("num_threads", num_threads_, 4);
    private_nh_.param<int>("min_points", min_points_, 80);
    private_nh_.param<double>("map_voxel", map_voxel_, 0.4);
    private_nh_.param<double>("scan_voxel", scan_voxel_, 0.3);
    private_nh_.param<double>("max_corr_dist", max_corr_dist_, 1.5);
    private_nh_.param<bool>("publish_tf", publish_tf_, false);

    reg_.setNumThreads(num_threads_);
    reg_.setRegistrationType("VGICP");
    reg_.setCorrespondenceRandomness(20);
    reg_.setMaxCorrespondenceDistance(max_corr_dist_);
    reg_.setVoxelResolution(1.0);

    LoadAndPrepareMap();

    cloud_sub_ = nh_.subscribe(input_cloud_topic_, 1, &SmallGicpRelocalizationNode::CloudCallback, this);
    initial_pose_sub_ = nh_.subscribe("/initialpose", 1, &SmallGicpRelocalizationNode::InitialPoseCallback, this);
    odom_pub_ = nh_.advertise<nav_msgs::Odometry>(output_odom_topic_, 10);

    ROS_INFO("small_gicp_relocalization_node started.");
    ROS_INFO("  map_pcd_path      : %s", map_pcd_path_.c_str());
    ROS_INFO("  input_cloud_topic : %s", input_cloud_topic_.c_str());
    ROS_INFO("  output_odom_topic : %s", output_odom_topic_.c_str());
    ROS_INFO("  map_frame/base    : %s -> %s", map_frame_.c_str(), base_frame_.c_str());
    ROS_INFO("  tf_child_frame    : %s", tf_child_frame_.c_str());
    ROS_INFO("  publish_tf        : %s", publish_tf_ ? "true" : "false");
  }

private:
  using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

  bool HasXYZFields(const sensor_msgs::PointCloud2& msg) {
    bool has_x = false;
    bool has_y = false;
    bool has_z = false;
    for (const auto& field : msg.fields) {
      if (field.name == "x") {
        has_x = true;
      } else if (field.name == "y") {
        has_y = true;
      } else if (field.name == "z") {
        has_z = true;
      }
    }
    return has_x && has_y && has_z;
  }

  CloudXYZ::Ptr VoxelDownsample(const CloudXYZ::ConstPtr& input, float leaf_size) {
    CloudXYZ::Ptr output(new CloudXYZ());
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input);
    voxel.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel.filter(*output);
    return output;
  }

  void LoadAndPrepareMap() {
    CloudXYZ::Ptr map_raw(new CloudXYZ());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_pcd_path_, *map_raw) != 0) {
      ROS_FATAL("Failed to load map pcd: %s", map_pcd_path_.c_str());
      ros::shutdown();
      return;
    }

    ROS_INFO("Loaded map raw points: %zu", map_raw->size());

    map_target_ = VoxelDownsample(map_raw, static_cast<float>(map_voxel_));
    reg_.setInputTarget(map_target_);

    ROS_INFO("Prepared map points after voxel: %zu", map_target_->size());
  }

  void InitialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {
    Eigen::Quaterniond quaternion(msg->pose.pose.orientation.w,
                                  msg->pose.pose.orientation.x,
                                  msg->pose.pose.orientation.y,
                                  msg->pose.pose.orientation.z);
    if (quaternion.norm() < 1e-9) {
      ROS_WARN("Received invalid initial pose quaternion, ignored.");
      return;
    }
    quaternion.normalize();

    Eigen::Isometry3d initial = Eigen::Isometry3d::Identity();
    initial.linear() = quaternion.toRotationMatrix();
    initial.translation() = Eigen::Vector3d(msg->pose.pose.position.x,
                                            msg->pose.pose.position.y,
                                            msg->pose.pose.position.z);

    std::lock_guard<std::mutex> lock(mutex_);
    latest_guess_ = initial;
    has_initial_guess_ = true;
    ROS_INFO("Initial pose received and set as matching guess.");
  }

  void CloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (!map_target_) {
      return;
    }

    if (msg->width == 0 || msg->height == 0 || msg->data.empty()) {
      ROS_WARN_THROTTLE(1.0, "Skip scan: empty PointCloud2 frame");
      return;
    }

    if (!HasXYZFields(*msg)) {
      ROS_WARN_THROTTLE(1.0, "Skip scan: PointCloud2 has no x/y/z fields");
      return;
    }

    CloudXYZ::Ptr scan_raw(new CloudXYZ());
    pcl::fromROSMsg(*msg, *scan_raw);
    if (static_cast<int>(scan_raw->size()) < min_points_) {
      ROS_WARN_THROTTLE(1.0, "Skip scan: raw points too few (%zu)", scan_raw->size());
      return;
    }

    CloudXYZ::Ptr scan_source = VoxelDownsample(scan_raw, static_cast<float>(scan_voxel_));
    if (static_cast<int>(scan_source->size()) < min_points_) {
      ROS_WARN_THROTTLE(1.0, "Skip scan: voxel points too few (%zu)", scan_source->size());
      return;
    }

    reg_.setInputSource(scan_source);

    Eigen::Isometry3d guess = Eigen::Isometry3d::Identity();
    bool has_guess = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      has_guess = has_initial_guess_;
      if (has_initial_guess_) {
        guess = latest_guess_;
      }
    }

    pcl::PointCloud<pcl::PointXYZ> aligned;
    if (has_guess) {
      reg_.align(aligned, guess.matrix().cast<float>());
    } else {
      reg_.align(aligned);
    }

    const Eigen::Matrix4f transform_f = reg_.getFinalTransformation();
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.matrix() = transform_f.cast<double>();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_guess_ = transform;
      has_initial_guess_ = true;
    }

    PublishPose(transform, msg->header.stamp);
  }

  void PublishPose(const Eigen::Isometry3d& transform, const ros::Time& stamp) {
    const Eigen::Quaterniond quaternion(transform.linear());
    const Eigen::Vector3d translation = transform.translation();

    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = translation.x();
    odom.pose.pose.position.y = translation.y();
    odom.pose.pose.position.z = translation.z();
    odom.pose.pose.orientation.x = quaternion.x();
    odom.pose.pose.orientation.y = quaternion.y();
    odom.pose.pose.orientation.z = quaternion.z();
    odom.pose.pose.orientation.w = quaternion.w();
    odom_pub_.publish(odom);

    ROS_INFO_THROTTLE(
      0.5,
      "Reloc pose [%s -> %s] t=(%.3f, %.3f, %.3f) q=(%.4f, %.4f, %.4f, %.4f)",
      map_frame_.c_str(),
      base_frame_.c_str(),
      translation.x(), translation.y(), translation.z(),
      quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w());

    if (publish_tf_) {
      geometry_msgs::TransformStamped tf_msg;
      tf_msg.header = odom.header;
      tf_msg.child_frame_id = tf_child_frame_;
      tf_msg.transform.translation.x = translation.x();
      tf_msg.transform.translation.y = translation.y();
      tf_msg.transform.translation.z = translation.z();
      tf_msg.transform.rotation.x = quaternion.x();
      tf_msg.transform.rotation.y = quaternion.y();
      tf_msg.transform.rotation.z = quaternion.z();
      tf_msg.transform.rotation.w = quaternion.w();
      tf_broadcaster_.sendTransform(tf_msg);
    }
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  ros::Subscriber cloud_sub_;
  ros::Subscriber initial_pose_sub_;
  ros::Publisher odom_pub_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  std::string map_pcd_path_;
  std::string input_cloud_topic_;
  std::string output_odom_topic_;
  std::string map_frame_;
  std::string base_frame_;
  std::string tf_child_frame_;

  int num_threads_;
  int min_points_;
  double map_voxel_;
  double scan_voxel_;
  double max_corr_dist_;
  bool publish_tf_;

  CloudXYZ::Ptr map_target_;
  small_gicp::RegistrationPCL<pcl::PointXYZ, pcl::PointXYZ> reg_;

  std::mutex mutex_;
  Eigen::Isometry3d latest_guess_ = Eigen::Isometry3d::Identity();
  bool has_initial_guess_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "small_gicp_relocalization_node");
  SmallGicpRelocalizationNode node;
  ros::spin();
  return 0;
}
