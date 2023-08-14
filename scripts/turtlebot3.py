import rospy
import math
import time

import numpy as np

from math import pi

from geometry_msgs.msg import (
    Twist,
    Point
)
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion

class PID:
	def __init__(self, 
					kp_linear = 0.1, kd_linear = 0.1, ki_linear = 0, 
					kp_angular = 0.1, kd_angular = 0.1, ki_angular = 0):
                
		self.__kp_linear = kp_linear
		self.__kd_linear = kd_linear
		self.__ki_linear = ki_linear

		self.__kp_angular = kp_angular
		self.__kd_angular = kd_angular
		self.__ki_angular = ki_angular

		self.__prev_error_position = 0
		self.__prev_error_angle = 0



	def get_control_inputs(self, distance_to_target, angle_to_target):
		error_position = distance_to_target
		
		error_angle = angle_to_target

		linear_velocity_control = self.__kp_linear*error_position + self.__kd_linear*(error_position - self.__prev_error_position)
		angular_velocity_control = self.__kp_angular*error_angle + self.__kd_angular*(error_angle - self.__prev_error_angle)

		self.prev_error_angle = error_angle
		self.prev_error_position = error_position

		return linear_velocity_control, angular_velocity_control



class TurtleBot3:
    def __init__(self):
        self._goal = Point()
        self._goal.x, self._goal.y = (0.6, 0.0)

        self.__position = Point()
        self.__heading = 0
        self.__yaw = 0
        self.__pid = PID()
        self.__cmd_vel = Twist()
        
        self.__raw_scan_range = []
        self.past_scan = {'left' : 3.5, 'forward' : 3.5, 'right' : 3.5, 'backward' : 3.5}
        self.current_scan = {'left' : 3.5, 'forward' : 3.5, 'right' : 3.5, 'backward' : 3.5}
        self.time_start = rospy.get_time()

        self.__read_parameters()
        self.__init_subscribers()
        self.__init_publishers()

    @property
    def goal(self):
        return self._goal
    
    @goal.setter
    def goal(self, pos):
        self._goal.x = pos[0]
        self._goal.y = pos[1]

    def __read_parameters(self) -> None:
        self.__subscriber_name_odom = rospy.get_param("turtlebot3/subscribers/odom")
        self.__subscriber_name_scan = rospy.get_param("turtlebot3/subscribers/scan")

        self.__publisher_name_cmd_vel = rospy.get_param("turtlebot3/publishers/cmd_vel")
        self.__publisher_name_debbug = rospy.get_param("turtlebot3/publishers/debbug")

    def __init_subscribers(self) -> None:
        rospy.Subscriber(self.__subscriber_name_odom, Odometry, \
                         self._odom_callback)
        
        rospy.Subscriber(self.__subscriber_name_scan, LaserScan, \
                         self._scan_callback, queue_size=1)
        
    def __init_publishers(self) -> None:
        self.__publisher_cmd_vel = rospy.Publisher(self.__publisher_name_cmd_vel, Twist, queue_size=5)
        self.__publisher_debbug = rospy.Publisher(self.__publisher_name_debbug, String, queue_size=5)

    def _odom_callback(self, odom) -> None:
        self.__position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.__yaw = yaw
                        
    def _scan_callback(self, scan):
        self.__raw_scan_range = scan.ranges

    def _heading(self, target= None, goal= []):
        if goal:
            goal_angle = math.atan2(goal[1] - self.__position.y, goal[0] - self.__position.x)

        else:
            goal_angle = target

        heading = goal_angle - self.__yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return round(heading, 2)
    
    def _euclidian_distance_to_target(self, target):
        target = np.array(target)
        position = np.array([self.__position.x, self.__position.y])

        return round(np.linalg.norm(target-position), 2)
    
    def _sample_adjust(self):
        try:
            self.past_scan = self.current_scan.copy()
            self.current_scan['left'] = min(self.__raw_scan_range[225:315])
            self.current_scan['forward'] = min(self.__raw_scan_range[315:359] + self.__raw_scan_range[0:45])
            self.current_scan['right'] = min(self.__raw_scan_range[45:135])
            self.current_scan['backward'] = min(self.__raw_scan_range[135:225])
        except:
            pass
        

    def stop(self):
        self.__publisher_cmd_vel.publish(Twist())

    def shutdown(self):
        rospy.loginfo("Stop...")
        self.stop()
        rospy.sleep(1)
        rospy.signal_shutdown("The End")    

    def collision_warn(self):
        
        left = True if self.current_scan['left'] <= self.past_scan['left'] else False
        forward = True if self.current_scan['forward'] <= self.past_scan['forward'] else False
        right = True if self.current_scan['right'] <= self.past_scan['right'] else False
        backward = True if self.current_scan['backward'] <= self.past_scan['backward'] else False
        return left, forward, right, backward
    
    def is_collision(self):
        return True if 0.13 > min(self.__raw_scan_range) > 0 else False

    def initial_euclidian_distance_to_goal(self):
        goal = np.array([self._goal.x, self._goal.y])
        position = np.array([-0.7, 0.0])

        return round(np.linalg.norm(goal-position), 2)

    def euclidian_distance_to_goal(self) -> float:
        goal = np.array([self._goal.x, self._goal.y])
        position = np.array([self.__position.x, self.__position.y])

        return round(np.linalg.norm(goal-position), 2)

    def get_state(self, action):
        self._sample_adjust()
        
        #rospy.logwarn(f"{self.current_scan['forward']-self.past_scan['forward']}")
        target_position = np.clip(action[0], -0.5, 0.5)
        target_angle = np.clip(action[1], -np.pi, np.pi)
        rospy.logwarn(f"Action ---> {action}")
        self.__cmd_vel.linear.x, self.__cmd_vel.angular.z = self.__pid.get_control_inputs(self._euclidian_distance_to_target(target_position),\
                                                                                          self._heading(target=target_angle))
        
        self.__publisher_cmd_vel.publish(self.__cmd_vel)
        
        distance_to_goal = self.euclidian_distance_to_goal()
        angle_to_goal = self._heading(goal=[self._goal.x, self._goal.y])

        left = self.current_scan['left']
        forward = self.current_scan['forward']
        right = self.current_scan['right'] 
        backward = self.current_scan['backward']

        done = True if distance_to_goal <= 0.2 else False

        rospy.logwarn(f"Action ---> {distance_to_goal, angle_to_goal} {self.__raw_scan_range}")

        observation = [distance_to_goal, angle_to_goal] + self.__raw_scan_range

        return observation, done
        