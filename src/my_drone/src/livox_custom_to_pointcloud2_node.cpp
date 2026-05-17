#include <ros/ros.h>

#include <livox_ros_driver2/CustomMsg.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

struct PointXYZIRT {
  PCL_ADD_POINT4D;
  PCL_ADD_INTENSITY;
  std::uint16_t ring;
  float time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIRT, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                    std::uint16_t, ring, ring)(float, time, time))

class LivoxCustomToPointCloud2Node {
 public:
  LivoxCustomToPointCloud2Node() : private_nh_("~") {
    private_nh_.param<std::string>("input_topic", input_topic_, "/livox/lidar");
    private_nh_.param<std::string>("output_topic", output_topic_, "/livox/lidar_pointcloud2");
    private_nh_.param<std::string>("frame_id", frame_id_, "");
    private_nh_.param<bool>("preserve_header_seq", preserve_header_seq_, true);

    pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_, 10);
    sub_ = nh_.subscribe(input_topic_, 10, &LivoxCustomToPointCloud2Node::Callback, this);

    ROS_INFO_STREAM("[livox_custom_to_pointcloud2] input_topic: " << input_topic_);
    ROS_INFO_STREAM("[livox_custom_to_pointcloud2] output_topic: " << output_topic_);
    if (!frame_id_.empty()) {
      ROS_INFO_STREAM("[livox_custom_to_pointcloud2] override frame_id: " << frame_id_);
    }
  }

 private:
  void Callback(const livox_ros_driver2::CustomMsg::ConstPtr& msg) {
    pcl::PointCloud<PointXYZIRT> cloud;
    cloud.reserve(msg->points.size());
    cloud.width = static_cast<std::uint32_t>(msg->points.size());
    cloud.height = 1;
    cloud.is_dense = true;

    for (const auto& point : msg->points) {
      PointXYZIRT pcl_point;
      pcl_point.x = point.x;
      pcl_point.y = point.y;
      pcl_point.z = point.z;
      pcl_point.intensity = static_cast<float>(point.reflectivity);
      pcl_point.ring = static_cast<std::uint16_t>(point.line);
      pcl_point.time = static_cast<float>(point.offset_time) * 1e-9f;
      cloud.push_back(pcl_point);
    }

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header = msg->header;
    if (!frame_id_.empty()) {
      cloud_msg.header.frame_id = frame_id_;
    }
    if (!preserve_header_seq_) {
      cloud_msg.header.seq = 0;
    }

    pub_.publish(cloud_msg);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  std::string input_topic_;
  std::string output_topic_;
  std::string frame_id_;
  bool preserve_header_seq_ = true;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "livox_custom_to_pointcloud2");
  LivoxCustomToPointCloud2Node node;
  ros::spin();
  return 0;
}
