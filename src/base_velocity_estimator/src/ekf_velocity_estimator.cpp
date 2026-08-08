#include "base_velocity_estimator/ekf_velocity_estimator.hpp"

EKFVelocityEstimator::EKFVelocityEstimator()
    : Node("base_velocity_estimator")
{   
    Q_.block<3,3>(3,3) *= 1e-5;  
    Q_.block<3,3>(6,6) *= 0.001;  

    bias_calibration_samples = 1000;
    H_(0, 0) = 1.0; // vx
    H_(1, 1) = 1.0; // vy
    H_(2, 2) = 1.0; // vz
    velocity_base_ = Eigen::Vector3d::Zero();
    accel_bias_ = Eigen::Vector3d::Zero();
    is_bias_calibrated_ = false;
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", 10, std::bind(&EKFVelocityEstimator::imuCallback, this, std::placeholders::_1));
    contact_sub = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/contact", 10, std::bind(&EKFVelocityEstimator::contactDetector, this, std::placeholders::_1));

    velocity_pub_ = create_publisher<geometry_msgs::msg::Twist>(
        "/base_velocity", 10);
    vel_obs_sub_ = create_subscription<geometry_msgs::msg::Twist>("/obs_velocity", 10,
                                                                  std::bind(&EKFVelocityEstimator::obsCallback, this, std::placeholders::_1));
    double publish_rate = 50;

    timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
        std::bind(&EKFVelocityEstimator::publishVelocity, this));
    RCLCPP_INFO(get_logger(), "Node initialized. Waiting for IMU data...");
}

void EKFVelocityEstimator::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    auto current_time = rclcpp::Time(msg->header.stamp);

    Eigen::Vector3d accel_measured = {
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z};
    if (!is_bias_calibrated_)
    {
        accel_samples_.push_back(accel_measured);
        if (accel_samples_.size() >= bias_calibration_samples)
        {
            calibrateBias();
            is_bias_calibrated_ = true;
            RCLCPP_INFO(get_logger(), "Bias calibration complete. Bias: [%f, %f, %f]",
                        accel_bias_.x(), accel_bias_.y(), accel_bias_.z());
        }
        return;
    }
    if (last_time_.nanoseconds() == 0)
    {
        last_time_ = current_time;
        return;
    }
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    Eigen::Vector3d accel_corrected = accel_measured - accel_bias_;
    // velocity_base_ += accel_corrected * dt;
    predict(accel_corrected, dt);
    accel_corrected.norm() > 0.1 ? small_acc_ = false : small_acc_ = true;
}
void EKFVelocityEstimator::contactDetector(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    float FL_force = msg->data[0];
    float FR_force = msg->data[1];
    float RL_force = msg->data[2];
    float RR_force = msg->data[3];
    updateForceBuffer(fl_force_buffer_, FL_force);
    updateForceBuffer(fr_force_buffer_, FR_force);
    updateForceBuffer(rl_force_buffer_, RL_force);
    updateForceBuffer(rr_force_buffer_, RR_force);
    bool legs_stable = isForceStable(fl_force_buffer_) && isForceStable(fr_force_buffer_) && isForceStable(rr_force_buffer_) && isForceStable(rl_force_buffer_);
    // RCLCPP_INFO(get_logger(), "FL: %.2f, FR: %.2f, RL: %.2f, RR: %.2f",
    // FL_force, FR_force, RL_force, RR_force);
    if (small_acc_ && legs_stable)
    {
        x_(0) = 0;
        x_(1) = 0;
        x_(2) = 0;
    }
}

void EKFVelocityEstimator::updateForceBuffer(std::deque<float> &force_buffer, float new_value)
{
    force_buffer.push_back(new_value);
    if (force_buffer.size() > window_size_)
        force_buffer.pop_front();
}
bool EKFVelocityEstimator::isForceStable(std::deque<float> &force_buffer)
{

    if (force_buffer.size() < window_size_)
        return false;
    float sq_sum = 0.0f;
    for (auto x : force_buffer)
    {
        sq_sum += x * x;
    }
    float mean = std::accumulate(force_buffer.begin(), force_buffer.end(), 0.0f) / window_size_;
    float stddev = std::sqrt(sq_sum / window_size_ - mean * mean);
    return stddev < stable_threshold_stddev;
}
void EKFVelocityEstimator::calibrateBias()
{
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto &sample : accel_samples_)
    {
        sum += sample;
    }
    accel_bias_ = sum / static_cast<double>(accel_samples_.size());
    accel_samples_.clear();
}

void EKFVelocityEstimator::predict(const Eigen::Vector3d &imu_acc, double dt)
{

    double b_ax = x_(6);
    double b_ay = x_(7);
    double b_az = x_(8);
    double ax = imu_acc(0) - b_ax;
    double ay = imu_acc(1) - b_ay;
    double az = imu_acc(2) - b_az;
    x_(0) += ax * dt;
    x_(1) += ay * dt;
    x_(2) += az * dt;

    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9);
    F(0, 3) = dt;  // vx to ax
    F(0, 6) = -dt; // vx to bax
    F(1, 4) = dt;  // vy to ay
    F(1, 7) = -dt; // vy to bay
    F(2, 5) = dt;  // vz to az
    F(2, 8) = -dt; // vz to baz
    P_ = F * P_ * F.transpose() + Q_;
}
void EKFVelocityEstimator::update(const Eigen::Vector3d &vel_obs)
{
    Eigen::Vector3d z = vel_obs;
    Eigen::Vector3d y = z - H_ * x_;

    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;

    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();

    x_ += K * y;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(9, 9);
    P_ = (I - K * H_) * P_;
}
void EKFVelocityEstimator::publishVelocity()
{
    if (!is_bias_calibrated_)
        return;

    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = x_(0);
    msg.linear.y = x_(1);
    msg.linear.z = x_(2);
    velocity_pub_->publish(msg);
}
void EKFVelocityEstimator::obsCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    Eigen::Vector3d obs_vel;
    obs_vel(0) = msg->linear.x;
    obs_vel(1) = msg->linear.y;
    obs_vel(2) = msg->linear.z;
    update(obs_vel);
}

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<EKFVelocityEstimator>();
//     rclcpp::spin(node);

//     rclcpp::shutdown();
//     return 0;
// }