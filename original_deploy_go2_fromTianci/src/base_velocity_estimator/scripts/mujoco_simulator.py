#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from pathlib import Path
from xbox_command import XboxController
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32MultiArray



class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator")
        self.cmd_sub = XboxController(self)  
        self.load_config()
        self.imu_pub=self.create_publisher(Imu,"/imu/data",10)
        self.contact_force_pub=self.create_publisher(Float32MultiArray,"/contact",10)
        self.joint_pub=self.create_publisher(Float32MultiArray,"/joint_angels",10)
        self.joint_vel_pub=self.create_publisher(Float32MultiArray,"/joint_velocities",10)
        self.omega_pub=self.create_publisher(Float32MultiArray,"gyro",10)
        self.z_axis_force_pub=self.create_publisher(Float32MultiArray,"/z_axis_force",10)
        self.step_counter = 0
        self.true_vel_pub=self.create_publisher(Float32MultiArray,"/true_velocities",10)

        # Initialize Mujoco
        self.init_mujoco()

        # Load policy
        self.policy = torch.jit.load(self.policy_path)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.timer = self.create_timer(self.simulation_dt, self.step_simulation)

    def load_config(self):
        current_file = Path(__file__).resolve()
        parent_dir = current_file.parent.parent
        config_file = parent_dir /'config'/'go2.yaml'
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            self.xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.lin_vel_scale = config["lin_vel_scale"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

    def init_mujoco(self):
        """Initialize Mujoco model and data"""
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[3:7]=[
              0.9932722449302673,
   0.008041736669838428,
   0.0063380408100783825,
   -0.11535090208053589]

        laydown=[-0.63224804,  0.8544461,  -2.7341957,   0.73985434,  0.9071291,  -2.6931078,
 -0.7145372,   0.43530887, -2.7260761,   0.76852196,  0.8470082,  -2.7184937 ]
        
        


        
        for i in range(12):
            self.d.qpos[7+i] =laydown[i] 
        self.m.opt.timestep = self.simulation_dt
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")
    def step_simulation(self):
        """Main simulation step (called by ROS 2 timer)"""
        # PD Control
        self.step_counter += 1

        tau = self.pd_control(
            self.target_dof_pos,
            self.d.qpos[7:],
            self.kps,
            self.d.qvel[6:],
            self.kds
        )
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        tau = [tau[index] for index in sequence]
        self.d.ctrl[:] = tau
        
        # Mujoco step
        mujoco.mj_step(self.m, self.d)
        
        # Publish sensor data
        self.publish_sensor_data()
        
        # Policy inference (every N steps)
        if self.step_counter % self.control_decimation ==0:
            self.run_policy()
        
        # Sync Mujoco viewer
        self.viewer.sync()

    def run_policy(self):
        """Run policy inference and update target DOF positions"""
        # Build observation vector
        self.cmd=np.zeros(3)
        # self.get_logger().info("run policy")
        self.left_button,self.right_button=self.cmd_sub.is_pressed()
        if self.left_button and self.right_button:
            linear_x,linear_y=self.cmd_sub.get_left_stick()
            angular_z=self.cmd_sub.get_right_stick()
            self.cmd=np.array([linear_x,linear_y,angular_z])
        # self.get_logger().info(f"FORCE {self.d.sensordata[55:]}")
        # print(len(self.d.sensordata))
        self.obs[:3] = self.d.sensordata[40:43] * self.ang_vel_scale  # Angular velocity
        self.obs[3:6] = self.get_gravity_orientation(self.d.qpos[3:7])  # Gravity vector
        self.obs[6:9] = self.cmd * self.cmd_scale  # Scaled command
        self.obs[9:21] = (self.d.qpos[7:19] - self.default_angles) * self.dof_pos_scale  # Joint positions
        self.obs[21:33] = self.d.qvel[6:18] * self.dof_vel_scale  # Joint velocities
        self.obs[33:45] = self.action  # Previous actions
        self.grav_acc=9.81*self.obs[3:6]
        # Policy inference
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
        print(self.target_dof_pos)
    def publish_sensor_data(self):
        imu_msg = Imu()
        if not hasattr(self,"grav_acc"): return
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.linear_acceleration.x = self.d.sensordata[43]+self.grav_acc[0]  # ax
        imu_msg.linear_acceleration.y = self.d.sensordata[44]+self.grav_acc[1]  # ay
        imu_msg.linear_acceleration.z = self.d.sensordata[45]+self.grav_acc[2]  # az
        self.imu_pub.publish(imu_msg)
        array=Float32MultiArray()
        fl_force_list=np.array([self.d.sensordata[i] for i in range (55,58)])
        fr_force_list=np.array([self.d.sensordata[i] for i in range (58,61)])
        rl_force_list=np.array([self.d.sensordata[i] for i in range (61,64)])
        rr_force_list=np.array([self.d.sensordata[i] for i in range (64,67)])
        FL_force=np.linalg.norm(fl_force_list)
        FR_force=np.linalg.norm(fr_force_list)
        RL_force=np.linalg.norm(rl_force_list)
        RR_force=np.linalg.norm(rr_force_list)
        array.data=[FL_force,FR_force,RL_force,RR_force]
        self.contact_force_pub.publish(array)
        angles_array=Float32MultiArray()
        angles_array.data=list(self.d.qpos[7:])
        self.joint_pub.publish(angles_array)
        q_dot_array=Float32MultiArray()
        q_dot_array.data=list(self.d.qvel[6:])
        self.joint_vel_pub.publish(q_dot_array)
        omega_array=Float32MultiArray()
        omega_array.data=list(self.d.sensordata[40:43])
        self.omega_pub.publish(omega_array)
        z_axis_force_array=Float32MultiArray()
        z_axis_force_array.data=[self.d.sensordata[i] for i in range (57,67,3)]
        # print([self.d.sensordata[i] for i in range (57,67,3)])
        self.z_axis_force_pub.publish(z_axis_force_array)
        true_velocity_array=Float32MultiArray()
        true_velocity_array.data=list(self.d.sensordata[52:55])
        self.true_vel_pub.publish(true_velocity_array)





    @staticmethod
    def get_gravity_orientation(quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        """Calculates torques from position commands"""
        torques=(target_q - q) * kp -  dq * kd
        return torques
    
    @staticmethod
    def quat_to_rot_matrix(q):
        """ 将四元数 (x, y, z, w) 转换为旋转矩阵 (3x3) """
        w,x, y, z = q
        R = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
        ])
        return R
    

def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

