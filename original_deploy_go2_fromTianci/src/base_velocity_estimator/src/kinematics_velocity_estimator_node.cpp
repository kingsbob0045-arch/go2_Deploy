
#include "base_velocity_estimator/kinematics_velocity_estimator_node.hpp"


KinematicsVelocityEstimator::KinematicsVelocityEstimator() : Node("Kinematics_Node")
{
    joint_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>("/joint_angels", 10,
                                                                       std::bind(&KinematicsVelocityEstimator::jointCallback, this, std::placeholders::_1));
    joint_vel_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/joint_velocities", 10, std::bind(&KinematicsVelocityEstimator::jointVelocityCallback, this, std::placeholders::_1));
    velocity_pub_ = create_publisher<geometry_msgs::msg::Twist>(
        "/obs_velocity", 10);
    omega_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/gyro", 10, std::bind(&KinematicsVelocityEstimator::angularVelocityCallback, this, std::placeholders::_1));
    contact_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/z_axis_force", 10, std::bind(&KinematicsVelocityEstimator::forceCallback, this, std::placeholders::_1));
    name_mapping_ = {{0, "FL_foot"}, {1, "FR_foot"}, {2, "RL_foot"}, {3, "RR_foot"}};
    std::vector<std::string> fl_group = {"FL_hip_joint",  "FL_thigh_joint","FL_calf_joint"};
    std::vector<std::string> fr_group = {"FR_hip_joint",  "FR_thigh_joint","FR_calf_joint"};
    std::vector<std::string> rl_group = {"RL_hip_joint",  "RL_thigh_joint","RL_calf_joint"};
    std::vector<std::string> rr_group = {"RR_hip_joint",  "RR_thigh_joint","RR_calf_joint"};
    joints_mapping_ = {{0, fl_group}, {1, fr_group}, {2, rl_group}, {3, rr_group}};
    timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_)),
        std::bind(&KinematicsVelocityEstimator::publishVelocity, this));

    // load Model from urdf
    std::string urdf_path = "/home/song/unitree_rl_gym/resources/robots/go2/urdf/go2.urdf";
    pinocchio::urdf::buildModel(urdf_path, model);
    data = Data(model);
    RCLCPP_INFO(get_logger(), "Node initialized. Waiting for data...");

}
void KinematicsVelocityEstimator::jointVelocityCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    cur_joint_velocities_ = std::vector<double>(msg->data.begin(), msg->data.end());
}
void KinematicsVelocityEstimator::jointCallback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    cur_joint_angles_ = std::vector<double>(msg->data.begin(), msg->data.end());
}
void KinematicsVelocityEstimator::angularVelocityCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    cur_omega_ = std::vector<double>(msg->data.begin(), msg->data.end());
}
void KinematicsVelocityEstimator::forceCallback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{

    legs_on_contact_.clear();
    for (int i = 0; i < 4; i++)
        if (abs(msg->data[i]) > force_threshold_)
            legs_on_contact_.push_back(i);
}

void KinematicsVelocityEstimator::publishVelocity()
{
    if (legs_on_contact_.empty() || cur_joint_angles_.empty() || cur_joint_velocities_.empty() || cur_omega_.empty())
    {
        if (legs_on_contact_.empty())
            RCLCPP_INFO(get_logger(), "short of legs on contact ");
        if (cur_joint_angles_.empty())
            RCLCPP_INFO(get_logger(), "short of cur_joint_angles_ ");
        if (cur_joint_velocities_.empty())
            RCLCPP_INFO(get_logger(), "short of cur_joint_velocities_ ");
        if (cur_omega_.empty())
            RCLCPP_INFO(get_logger(), "short of cur_omega_ ");
        return;
    }
    // RCLCPP_INFO(get_logger(), "proceeding");
    std::vector<int> legs_indices=legs_on_contact_;
    std::string frame_name;
    std::vector<std::string> joint_names;
    std::vector<double> theta_f = cur_joint_angles_;
    std::vector<double> theta_dot = cur_joint_velocities_;
    std::vector<double> omega_f = cur_omega_;
    int n=legs_indices.size();
    Eigen::Matrix3Xd matrix(3,n);
    std::vector<Eigen::Vector3d> vectors;
    for (int leg_index : legs_indices)
    {
        frame_name = name_mapping_[leg_index];
        joint_names = joints_mapping_[leg_index];
        Eigen::VectorXd theta = Eigen::Map<Eigen::VectorXd>(theta_f.data(), theta_f.size());
        Eigen::Vector3d omega = Eigen::Map<Eigen::Vector3d>(omega_f.data());
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> pair = compute_kinematics_and_jacobian(theta, frame_name, joint_names);
        Eigen::Vector3d position = pair.first;
        Eigen::Matrix3d jaccobian = pair.second;
        std::vector<double> q_dot_f(
            theta_dot.begin() + leg_index * 3,
            theta_dot.begin() + (leg_index + 1) * 3);
        Eigen::Vector3d q_dot = Eigen::Map<Eigen::Vector3d>(q_dot_f.data());

        Eigen::Vector3d v_base = -omega.cross(position) - jaccobian * q_dot;
        vectors.push_back(v_base);
        // std::cout << "v base is:" << v_base << std::endl;
    }
    for (size_t i = 0;i<n; i++) {
        matrix.col(i) = vectors[i];
    }
    Eigen::Vector3d velocity_kinematics_=matrix.rowwise().mean();
    // velocity_kinematics_=filter.filter(velocity_kinematics_);
    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = velocity_kinematics_.x();
    msg.linear.y = velocity_kinematics_.y();
    msg.linear.z = velocity_kinematics_.z();
    velocity_pub_->publish(msg);
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> KinematicsVelocityEstimator::compute_kinematics_and_jacobian(Eigen::VectorXd &theta,
                                                                                                         const std::string &frame_name, const std::vector<std::string> &joint_names)
{

    Eigen::VectorXd q = theta; // zero Init

    // compute link's position related to base
    int frame_id = model.getFrameId(frame_name); // get Link ID
    // std::cout<<"link_id:"<<frame_id<<std::endl;
    pinocchio::forwardKinematics(model, data, q); // update fK
    pinocchio::updateFramePlacements(model, data);
    pinocchio::SE3 T_base_link = data.oMf[frame_id]; // Transformation Matrix
    // extract Position and Orientation
    Eigen::Vector3d position = T_base_link.translation();

    // get Joint Order
    // for (pinocchio::JointIndex joint_id = 1; joint_id < model.joints.size(); ++joint_id)
    // {
    //     const auto &joint = model.joints[joint_id];
    //     const std::string &name = model.names[joint_id];
    //     int nq = joint.nq(); // 当前关节在q中的维度

    //     std::cout << "Joint '" << name << "' (id=" << joint_id
    //               << ") occupies " << nq << " values in q: ";
    // }

    // compute Jaccobian Matrix
    Eigen::MatrixXd J(6, model.nv); // 6xn Matrix
    computeFrameJacobian(model, data, q, frame_id, LOCAL_WORLD_ALIGNED, J);
    std::vector<int> indices;
    // get joint indices
    for (const auto &name : joint_names)
    {
        JointIndex jid = model.getJointId(name);
        indices.push_back(model.joints[jid].idx_v());
    }
    Eigen::Matrix3d J_linear;
    // extract relevant Entries, rows[0:0+3] stands for linear velocity, indices[x:x+1] only cares for the related joint
    for (int i = 0; i < 3; i++)
        J_linear.col(i) = J.block(0, indices[i], 3, 1);
    // std::cout << J_linear;
    return std::pair<Eigen::Vector3d, Eigen::Matrix3d>{position, J_linear};
}

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<KinematicsVelocityEstimator>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }