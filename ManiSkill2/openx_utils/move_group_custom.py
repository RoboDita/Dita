from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import moveit_commander
import rospy
import sys

class MoveGroup:
    def __init__(self,
                 planner: str = "PTP",
                 pose_ref_frame: str = "panda_link0",
                 allow_replanning: bool = False,
                 planning_attempts: int = 100,
                 planning_time: float = 2.6,
                 goal_position_tolerance: float = 0.0001,
                 goal_orientation_tolerance: float = 0.001
                 ):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_custom", anonymous=True)
        self.current_joints: list = []
        self.goal_position_tolerance = goal_position_tolerance
        self.goal_orientation_tolerance = goal_orientation_tolerance
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.allow_replanning = allow_replanning
        self.pose_ref_frame = pose_ref_frame
        self.planner = planner
        self.move_group = moveit_commander.MoveGroupCommander("panda_manipulator")
        self.grasping_group = moveit_commander.MoveGroupCommander("panda_hand")
        self.current_pose = self.get_current_pose()
        self.move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
        self.target_pose_publisher = rospy.Publisher("/move_group_custom/target_pose", PoseStamped, queue_size=20)

    def get_current_pose(self) -> PoseStamped:
        """
        Return:
            PoseStamped object with position(x, y, z)
            and orientation(x, y, z, w)
        """
        return self.move_group.get_current_pose()

    def get_current_joints(self) -> list:
        """
        Return:
            A list with the values of the joints angles in radian
        """
        return self.grasping_group.get_current_joint_values()

    def interface(self, pose: list, gripper: float, sync=True):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "panda_link0"
        pose_goal.pose.position.x = float(pose[0])
        pose_goal.pose.position.y = float(pose[1])
        pose_goal.pose.position.z = float(pose[2])
        pose_goal.pose.orientation.w = float(pose[3])
        pose_goal.pose.orientation.x = float(pose[4])
        pose_goal.pose.orientation.y = float(pose[5])
        pose_goal.pose.orientation.z = float(pose[6])
        self.target_pose_publisher.publish(pose_goal)

        gripper_joint_max, gripper_cmd_max = 0.034928795419214294, 1.0 # read from moveit panel
        gripper_joint_min, gripper_cmd_min = 9.407789388205857e-05, -1.0
        D, d = gripper_joint_max - gripper_joint_min, gripper_cmd_max - gripper_cmd_min
        joint_val = gripper_joint_min + (gripper - gripper_cmd_min) * D / d

        if sync:
            self.plan_and_execute_pose_with_gripper(pose=pose_goal, gripper=joint_val)
        else:
            self.plan_and_execute_pose(pose=pose_goal, planner="PTP")
            joint_goal = self.get_current_joints()
            joint_goal[0], joint_goal[1] = joint_val, joint_val
            self.plan_and_execute_joints(joints=joint_goal)

    def plan_and_execute_pose_with_gripper(self, pose: PoseStamped, gripper: float):
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose)
        self.set_planning_config()
        plans = self.move_group.plan()
        if not MoveGroup.plan_is_successful(plans):
            return

        plan = plans[1]
        plan.joint_trajectory.joint_names.append("panda_finger_joint1")
        plan.joint_trajectory.joint_names.append("panda_finger_joint2")
        points_num = len(plan.joint_trajectory.points)
        jv_0 = self.move_group.get_current_state().joint_state.position[7] # panda_finger_joint1
        jv_d = (gripper - jv_0) / points_num
        for i in range(points_num):
            jv = jv_0 + i * jv_d
            plan.joint_trajectory.points[i].positions = plan.joint_trajectory.points[i].positions + (jv, jv)
            plan.joint_trajectory.points[i].velocities = plan.joint_trajectory.points[i].velocities + (0.0, 0.0)
            plan.joint_trajectory.points[i].accelerations = plan.joint_trajectory.points[i].accelerations + (0.0, 0.0)

        success = self.move_group.execute(plan, wait=True)
        if not success:
            return

        self.move_group.stop()

    def plan_and_execute_pose(self, pose: PoseStamped, planner: str = 'PTP') -> any:
        """
        Plan the path from start state to goal state. These states
        are set as a pose represented by a PoseStamped object


        Args:
            pose: PoseStamped object with position(x, y, z)
             and orientation(x, y, z, w)
                OBS: Supply angles in degrees

            planner: A string with the planner to use. The name needs
                     to be the same as in Rviz grapical interface.
                     Ex: 'RRTConnect'

        Return:
            ...
        """
        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(pose)
        self.set_planning_config(planner=planner)
        plan = self.move_group.plan()
        if not MoveGroup.plan_is_successful(plan):
            return

        success = self.move_group.execute(plan[1], wait=True)
        if not success:
            return

        self.current_pose = self.move_group.get_current_pose()
        return plan[1]

    def plan_and_execute_joints(self, joints: list, planner: str = 'PTP'):
        """
        Plan the path from start state to goal state. These states
        are set as joints angles.


        Args:
            joints: A list of robot's joints angles with degrees values
                OBS: Supply angles in degrees

            planner: A string with the planner to use. The name needs
                     to be the same as in Rviz grapical interface.
                     Ex: 'LazyPRMstar'

        Return:
            ...
        """
        self.grasping_group.set_start_state_to_current_state()
        self.grasping_group.set_joint_value_target(joints)
        # self.set_planning_config(planner=planner)
        plan = self.grasping_group.plan()
        if not MoveGroup.plan_is_successful(plan):
            return

        success = self.grasping_group.execute(plan[1], wait=True)
        if not success:
            return

        self.current_joints = self.grasping_group.get_current_joint_values()

    def is_same_pose(self, pose_start, pose_goal):
        ...

    def set_planning_config(self,
                            planner: str = "PTP",
                            pose_ref_frame: str = "panda_link0",
                            allow_replanning: bool = False,
                            planning_attempts: int = 50,
                            planning_time: float = 3.0,
                            goal_position_tolerance: float = 0.0001,
                            goal_orientation_tolerance: float = 0.001
                            ):
        """
        Set configuration to use in planning scenes. Such as:

        planner: Planner to be used. Set it as a string
        equal to the name in rviz graphical interface

        pose_ref_frame: Reference frame inside PoseStamped object

        planning_attempts: Tries till timeout

        planning_time: Time till timeout. The timeout occurs when planning_attempts or planning time is reached

        goal_xxx_tolerance: Helps when planning is consistently failing, but raises collision probability

        allow_replanning:

        Default configuration:

        planner: str = "RRTConnect",
        pose_ref_frame: str = "panda_link0",
        allow_replanning: bool = False,
        planning_attempts: int = 50,
        planning_time: float = 3.0,
        goal_position_tolerance: float = 0.0001,
        oal_orientation_tolerance: float = 0.001

        """
        self.planning_time = planning_time
        self.planning_attempts = planning_attempts
        self.allow_replanning = allow_replanning
        self.pose_ref_frame = pose_ref_frame
        self.planner = planner

        self.move_group.set_planner_id(self.planner)
        self.move_group.set_pose_reference_frame(self.pose_ref_frame)
        self.move_group.allow_replanning(self.allow_replanning)
        self.move_group.set_num_planning_attempts(self.planning_attempts)
        self.move_group.set_planning_time(self.planning_time)
        self.move_group.set_goal_position_tolerance(goal_position_tolerance)
        self.move_group.set_goal_orientation_tolerance(goal_orientation_tolerance)

        self.grasping_group.set_planner_id(self.planner)

    @staticmethod
    def plan_is_successful(plan: tuple):
        """
        Args:
             plan (tuple): A tuple with the following elements:
                (MoveItErrorCodes, trajectory_msg, planning_time, error_code)

        Returns:
            bool: True if plan successfully computed.
        """

        # print("plan success", plan[0])
        return plan[0]