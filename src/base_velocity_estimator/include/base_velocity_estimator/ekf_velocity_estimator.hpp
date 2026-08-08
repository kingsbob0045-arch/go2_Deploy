#pragma once
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <eigen3/Eigen/Dense>
#include <deque>
class EKFVelocityEstimator : public rclcpp::Node
{
public:
    EKFVelocityEstimator();
    void predict(const Eigen::Vector3d &imu_acc, double dt);
    void update(const Eigen::Vector3d &vel_obs);
    // ~imuVelocityEstimator();
private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void obsCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void calibrateBias();
    void publishVelocity();
    void contactDetector(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void updateForceBuffer(std::deque<float> &force_buffer, float new_value);
    bool isForceStable(std::deque<float> &force_buffer_);
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_obs_sub_;

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr contact_sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    Eigen::Vector3d velocity_base_;
    Eigen::Vector3d accel_bias_;
    std::deque<Eigen::Vector3d> accel_samples_;
    bool is_bias_calibrated_ = false;
    rclcpp::Time last_time_;
    int bias_calibration_samples = 1000;
    bool small_acc_ = true;
    const int window_size_ = 20;
    const float stable_threshold_stddev = 0.1;
    std::deque<float> fl_force_buffer_;
    std::deque<float> rl_force_buffer_;
    std::deque<float> fr_force_buffer_;
    std::deque<float> rr_force_buffer_;
    Eigen::VectorXd x_ = Eigen::VectorXd::Zero(9);              // 状态向量: [vx, vy, vz, ax, ay, az, bax, bay, baz]
    Eigen::MatrixXd P_ = Eigen::MatrixXd::Identity(9, 9) * 5; // 协方差矩阵
    Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(9, 9)*0.01;           // 过程噪声协方差

    // 观测噪声协方差
    Eigen::MatrixXd R_ = Eigen::MatrixXd::Identity(3, 3) * 10;
    Eigen::MatrixXd H_ = Eigen::MatrixXd::Zero(3, 9); // 观测矩阵
};