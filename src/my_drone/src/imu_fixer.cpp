#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <fstream>
#include <string>
#include <cmath>

class IMUFixerNode
{
public:
  IMUFixerNode()
  {
    ros::NodeHandle private_nh("~");
    private_nh.param<std::string>("input_topic", input_topic_, "/livox/imu");
    private_nh.param<std::string>("output_topic", output_topic_, "/livox/imu2");
    private_nh.param<std::string>("config_path", config_path_, "");
    private_nh.param<double>("pitch", pitch_, 0.0);
    private_nh.param<double>("roll", roll_, 0.0);
    private_nh.param<double>("yaw", yaw_, 0.0);
    private_nh.param<double>("acc_scale", acc_scale_, 9.80511);
    private_nh.param<bool>("rotate_orientation", rotate_orientation_, false);

    if (pitch_ == 0.0 && roll_ == 0.0 && yaw_ == 0.0) {
      if (!ReadExtrinsic()) {
        ROS_WARN("Failed to read extrinsic parameters from JSON, using zeros.");
      } else {
        ROS_INFO("Read extrinsic from JSON: roll=%.3f pitch=%.3f yaw=%.3f", roll_, pitch_, yaw_);
      }
    } else {
      ROS_INFO("Using ROS params: roll=%.3f pitch=%.3f yaw=%.3f", roll_, pitch_, yaw_);
    }

    tf2::Quaternion q;
    q.setRPY(deg2rad(roll_), deg2rad(pitch_), deg2rad(yaw_));
    q.normalize();
    rotation_q_ = q;

    pub_ = nh_.advertise<sensor_msgs::Imu>(output_topic_, 100);
    sub_ = nh_.subscribe(input_topic_, 100, &IMUFixerNode::ImuCallback, this);
    ROS_INFO("Subscribed to %s, publishing %s, acc_scale=%.5f, rotate_orientation=%s",
             input_topic_.c_str(), output_topic_.c_str(), acc_scale_,
             rotate_orientation_ ? "true" : "false");
  }

private:
  bool ReadExtrinsic()
  {
    if (config_path_.empty()) {
      ROS_WARN("config_path is empty, skip reading extrinsic config.");
      return false;
    }

    std::ifstream ifs(config_path_);
    if (!ifs.is_open()) {
      ROS_WARN("Cannot open config at %s", config_path_.c_str());
      return false;
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    auto ext_pos = content.find("\"extrinsic_parameter\"");
    if (ext_pos == std::string::npos) {
      ROS_WARN("extrinsic_parameter missing");
      return false;
    }
    auto brace_open = content.find('{', ext_pos);
    if (brace_open == std::string::npos) return false;
    auto brace_close = content.find('}', brace_open);
    if (brace_close == std::string::npos) return false;
    std::string block = content.substr(brace_open, brace_close - brace_open + 1);

    auto extract = [&](const std::string &s, const std::string &key, double &out)->bool {
      std::string pattern = "\"" + key + "\"";
      auto pos = s.find(pattern);
      if (pos == std::string::npos) return false;
      pos = s.find(':', pos);
      if (pos == std::string::npos) return false;
      pos++;
      while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
      size_t end = pos;
      bool found = false;
      while (end < s.size() && ( (s[end] >= '0' && s[end] <= '9') || s[end]=='-' || s[end]=='+' || s[end]=='.' || s[end]=='e' || s[end]=='E')) { end++; found = true; }
      if (!found) return false;
      try {
        out = std::stod(s.substr(pos, end - pos));
        return true;
      } catch (...) {
        return false;
      }
    };

    extract(block, "roll", roll_);
    extract(block, "pitch", pitch_);
    extract(block, "yaw", yaw_);
    extract(block, "x", x_);
    extract(block, "y", y_);
    extract(block, "z", z_);

    return true;
  }

  void ImuCallback(const sensor_msgs::Imu::ConstPtr& msg)
  {
    sensor_msgs::Imu out = *msg;

    tf2::Vector3 acc(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    
    // Livox MID360 acceleration is published in g by livox_ros_driver2.
    acc = acc * acc_scale_;

    tf2::Vector3 ang(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

    tf2::Matrix3x3 R(rotation_q_);
    tf2::Vector3 acc_r = R * acc;
    tf2::Vector3 ang_r = R * ang;

    out.linear_acceleration.x = acc_r.x();
    out.linear_acceleration.y = acc_r.y();
    out.linear_acceleration.z = acc_r.z();

    out.angular_velocity.x = ang_r.x();
    out.angular_velocity.y = ang_r.y();
    out.angular_velocity.z = ang_r.z();

    tf2::Quaternion q_out(0.0, 0.0, 0.0, 1.0);
    const double q_norm_sq = out.orientation.x * out.orientation.x +
                             out.orientation.y * out.orientation.y +
                             out.orientation.z * out.orientation.z +
                             out.orientation.w * out.orientation.w;
    if (std::isfinite(q_norm_sq) && q_norm_sq > 1e-12) {
      tf2::Quaternion q_orig(out.orientation.x, out.orientation.y,
                             out.orientation.z, out.orientation.w);
      q_out = rotate_orientation_ ? rotation_q_ * q_orig : q_orig;
      q_out.normalize();
    }
    out.orientation.x = q_out.x();
    out.orientation.y = q_out.y();
    out.orientation.z = q_out.z();
    out.orientation.w = q_out.w();

    pub_.publish(out);
  }

  double deg2rad(double d) { return d * M_PI / 180.0; }

  std::string input_topic_;
  std::string output_topic_;
  std::string config_path_;
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;

  // extrinsic
  double roll_{0.0}, pitch_{0.0}, yaw_{0.0}, x_{0.0}, y_{0.0}, z_{0.0};
  double acc_scale_{9.80511};
  bool rotate_orientation_{false};
  tf2::Quaternion rotation_q_;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "imu_fixer_node");
  IMUFixerNode node;
  ros::spin();
  return 0;
}
