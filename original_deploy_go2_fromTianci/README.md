# Deploy-an-RL-policy-on-the-Unitree-Go2-robot
This repository provides a framework for low-level control of a legged robot (Unitree Go2), using ROS 2 as the communication middleware. The MuJoCo simulator is used to validate the control policy in simulation. Once the policy performs well in MuJoCo, you can deploy it on the real robot by simply setting the ROS parameter is_simulation to false. Also a base velocity estimator using extended Karman Filter is provided to estimate the velocity of base. If you find this project useful, please consider giving it a ⭐️ to support development!


## Environment
- **Ubuntu**: 20.04/22.04
- **ROS 2**: Foxy/Humble
- **MuJoCo**: 3.2.3
- **Python**: 3.8.20/3.10
- **Pinocchio**: 3.4.0

## Access Robot Sensor Data via ROS 2
Refer to the official Unitree ROS 2 repository for setup and examples:
[unitree ros2](https://github.com/unitreerobotics/unitree_ros2)

*source unitree_ros2 before building this workspace*

### Potential Build Issue
When building the ROS 2 packages, you may encounter the following error:
```bash
ModuleNotFoundError: No module named 'unitree_go.unitree_go_s__rosidl_typesupport_c’
```
or 
``` bash
rosidl_generator_py.import_type_support_impl.UnsupportedTypeSupport: Could not import 'rosidl_typesupport_c' for package 'unitree_go’
```
This issue occurs because **ROS 2 Foxy supports only Python 3.8** for generating C-based type support modules. If a different Python version is used, the build process may still succeed, but the runtime will **fail to locate the generated files**, resulting in import errors when executing ROS 2 commands.

### Solution
Manually set your Python version to **3.8** when building the workspace. For example:
```bash
export PYTHON_EXECUTABLE=/usr/bin/python3.8
colcon build --symlink-install
```
Make sure Python 3.8 is installed and available at the specified path.




## Simulation
After successfully building this workspace, you can launch the simulation in mujoco with the following commands:
```bash
source install/setup.bash
ros2 run deploy_rl_policy mujoco_simulator.py
``` 
Here is a screenshot of the simulation scene:
<p align="center">
  <img src="./resources/images/mujoco.png" alt="MuJoCo Simulation Scene" width="500"/>
</p>

## Control Logic
The robot's behavior is controlled by a 3-state state machine, including:
* **Laying Down**
* **Standing Up**
* **Executing RL Policy**

### State Transition (XBox Controller)
* **Initial State:** Robot automatically enters "Laying Down" state
* **B Button:** Transitions from "Laying Down" → "Standing Up"
* **A Button:** Transitions from "Standing Up" → "Laying Down"
* **LB + RB Simultaneously:** While standing, executes RL Policy (remains in standing state)

### Notes:
* The "Executing RL Policy" state is considered a special case of the "Standing Up" state
* Controller inputs are only processed when the robot is in the appropriate state for that transition

### Launch Control Nodes
**Remember to turn down the *sport mode* service of the robot before deploying the policy on real robot!**

Run the following commands in separate terminals to activate the control system:

```bash
# Terminal 1: XBox Controller Interface
ros2 run joy joy_node

# Terminal 2: State Machine Controller
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true # true: simulation  false: real robot

# Terminal 3: Reinforcement Learning Policy
ros2 run deploy_rl_policy rl_policy.py --is_simulation True  # or False
```
Node Description:
1. joy_node
    * Interfaces with XBox controller hardware
    * Publishes controller input to /joy topic
2. low_level_control
    * Implements the 3-state machine (Laying Down/Standing Up/RL Policy)
    * Handles state transitions based on controller input
    * Sends lowcmd to simulator or the real robot
3. ​​RL_policy.py​​:
    * Executes reinforcement learning policy
    * Activated only in "Executing RL Policy" state (LB+RB pressed while standing)

## Implementing Custom Policies

To use your own reinforcement learning policy with the system:

1. **Modify Policy Path**  
   Edit the policy file path in `rl_policy.py` to point to your custom policy.

2. **Data Sequence Considerations**  
   - The Unitree robot uses a specific joint order that may differ from your training environment
   - Verify your policy's output sequence matches the robot's expected input order

3. **Safety Recommendations**  
   ```diff
   + Always test new policies in simulation first
   - Avoid deploying untested policies directly to hardware
   ```
You can refer to the [official documentation](https://support.unitree.com/home/en/developer/Basic_services) to check the correct joint order.

## Base Velocity Estimator
If your policy requires the base velocity as part of the observation and it's not available from onboard sensors, you can use the **Base Velocity Estimator** to estimate it.

It's implemented as an **Extended Kalman Filter (EKF)** with a measurement model and a system model. The measurement is computed from kinematic equations using [Pinocchio](https://github.com/stack-of-tasks/pinocchio). Here's a rough overview of the theory behind it:
[https://glowing-torch.github.io/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/](https://glowing-torch.github.io/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/).
