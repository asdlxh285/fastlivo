#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <mutex>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cmath>

class PointCloudFusionNode
{
public:
  PointCloudFusionNode()
    : private_nh_("~"),
      tf_buffer_(),
      tf_listener_(tf_buffer_)
  {
    private_nh_.param<std::string>("lidar_topic", lidar_topic_, "/livox/lidar");
    private_nh_.param<std::string>("camera_topic", camera_topic_, "/camera/depth/points");
    private_nh_.param<std::string>("output_topic", output_topic_, "/fusion/points");
    private_nh_.param<std::string>("target_frame", target_frame_, "");
    private_nh_.param<double>("camera_cache_timeout", camera_cache_timeout_, 0.5);
    private_nh_.param<double>("camera_intensity", camera_intensity_, 20.0);
    private_nh_.param<bool>("allow_untransformed_fallback", allow_untransformed_fallback_, true);

    lidar_sub_ = nh_.subscribe(lidar_topic_, 10, &PointCloudFusionNode::LidarCallback, this);
    camera_sub_ = nh_.subscribe(camera_topic_, 10, &PointCloudFusionNode::CameraCallback, this);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 10);

    ROS_INFO("PointCloudFusionNode started.");
    ROS_INFO("  lidar_topic   : %s", lidar_topic_.c_str());
    ROS_INFO("  camera_topic  : %s", camera_topic_.c_str());
    ROS_INFO("  output_topic  : %s", output_topic_.c_str());
    ROS_INFO("  target_frame  : %s", target_frame_.empty() ? "<use lidar frame>" : target_frame_.c_str());
    ROS_INFO("  camera_cache_timeout(sec): %.3f", camera_cache_timeout_);
    ROS_INFO("  allow_untransformed_fallback: %s", allow_untransformed_fallback_ ? "true" : "false");
  }

private:
  bool TransformCloudToFrame(const pcl::PointCloud<pcl::PointXYZI>& in,
                             const std::string& from_frame,
                             const ros::Time& stamp,
                             const std::string& to_frame,
                             pcl::PointCloud<pcl::PointXYZI>& out)
  {
    if (from_frame == to_frame) {
      out = in;
      return true;
    }

    try {
      geometry_msgs::TransformStamped transform =
          tf_buffer_.lookupTransform(to_frame, from_frame, stamp, ros::Duration(0.05));

      const auto& t = transform.transform.translation;
      const auto& q = transform.transform.rotation;
      Eigen::Translation3f translation(static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z));
      Eigen::Quaternionf rotation(static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y), static_cast<float>(q.z));
      Eigen::Matrix4f tf_matrix = (translation * rotation).matrix();

      pcl::transformPointCloud(in, out, tf_matrix);
      return true;
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(1.0, "TF transform failed (%s -> %s): %s", from_frame.c_str(), to_frame.c_str(), ex.what());
      return false;
    }
  }

  void ConvertToXYZI(const sensor_msgs::PointCloud2& cloud_msg,
                     pcl::PointCloud<pcl::PointXYZI>& out,
                     bool overwrite_intensity,
                     float intensity_value)
  {
    out.clear();
    out.reserve(static_cast<size_t>(cloud_msg.width) * static_cast<size_t>(cloud_msg.height));

    bool has_intensity = false;
    for (const auto& field : cloud_msg.fields) {
      if (field.name == "intensity") {
        has_intensity = true;
        break;
      }
    }

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud_msg, "z");

    if (has_intensity && !overwrite_intensity) {
      sensor_msgs::PointCloud2ConstIterator<float> iter_i(cloud_msg, "intensity");
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
          continue;
        }
        pcl::PointXYZI p;
        p.x = *iter_x;
        p.y = *iter_y;
        p.z = *iter_z;
        p.intensity = *iter_i;
        out.push_back(p);
      }
    } else {
      for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
          continue;
        }
        pcl::PointXYZI p;
        p.x = *iter_x;
        p.y = *iter_y;
        p.z = *iter_z;
        p.intensity = intensity_value;
        out.push_back(p);
      }
    }

    out.width = static_cast<uint32_t>(out.size());
    out.height = 1;
    out.is_dense = false;
  }

  void CameraCallback(const sensor_msgs::PointCloud2::ConstPtr& camera_msg)
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    latest_camera_msg_ = camera_msg;
    latest_camera_arrival_ = ros::Time::now();
  }

  void LidarCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg)
  {
    sensor_msgs::PointCloud2::ConstPtr camera_msg;
    ros::Time camera_arrival;
    {
      std::lock_guard<std::mutex> lock(camera_mutex_);
      camera_msg = latest_camera_msg_;
      camera_arrival = latest_camera_arrival_;
    }

    if (!camera_msg) {
      ROS_WARN_THROTTLE(1.0, "Waiting for first camera cloud on %s", camera_topic_.c_str());
      return;
    }

    if ((ros::Time::now() - camera_arrival).toSec() > camera_cache_timeout_) {
      ROS_WARN_THROTTLE(1.0, "Latest camera cloud too old (%.3fs), skip fusion", (ros::Time::now() - camera_arrival).toSec());
      return;
    }

    const std::string output_frame = target_frame_.empty() ? lidar_msg->header.frame_id : target_frame_;

    pcl::PointCloud<pcl::PointXYZI> lidar_cloud;
    pcl::PointCloud<pcl::PointXYZI> camera_cloud;
    ConvertToXYZI(*lidar_msg, lidar_cloud, false, 0.0f);
    ConvertToXYZI(*camera_msg, camera_cloud, true, static_cast<float>(camera_intensity_));

    pcl::PointCloud<pcl::PointXYZI> lidar_in_target;
    pcl::PointCloud<pcl::PointXYZI> camera_in_target;

    if (!TransformCloudToFrame(lidar_cloud, lidar_msg->header.frame_id, lidar_msg->header.stamp, output_frame, lidar_in_target)) {
      return;
    }
    if (!TransformCloudToFrame(camera_cloud, camera_msg->header.frame_id, camera_msg->header.stamp, output_frame, camera_in_target)) {
      if (!allow_untransformed_fallback_) {
        return;
      }
      camera_in_target = camera_cloud;
      ROS_WARN_THROTTLE(1.0, "Using untransformed camera cloud fallback; please provide TF from %s to %s for accurate fusion.",
                        camera_msg->header.frame_id.c_str(), output_frame.c_str());
    }

    pcl::PointCloud<pcl::PointXYZI> fused_cloud = lidar_in_target;
    fused_cloud += camera_in_target;

    sensor_msgs::PointCloud2 fused_msg;
    pcl::toROSMsg(fused_cloud, fused_msg);
    fused_msg.header.stamp = lidar_msg->header.stamp;
    fused_msg.header.frame_id = output_frame;

    pub_.publish(fused_msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  std::string lidar_topic_;
  std::string camera_topic_;
  std::string output_topic_;
  std::string target_frame_;
  double camera_cache_timeout_{0.5};
  double camera_intensity_{20.0};
  bool allow_untransformed_fallback_{true};

  ros::Publisher pub_;
  ros::Subscriber lidar_sub_;
  ros::Subscriber camera_sub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex camera_mutex_;
  sensor_msgs::PointCloud2::ConstPtr latest_camera_msg_;
  ros::Time latest_camera_arrival_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_fusion_node");
  PointCloudFusionNode node;
  ros::spin();
  return 0;
}
