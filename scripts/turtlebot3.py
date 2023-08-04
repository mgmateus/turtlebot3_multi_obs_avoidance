import rospy
import math

import numpy as np

from math import pi

from geometry_msgs.msg import (
    Twist,
    Pose
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion




class TurtleBot3:
    def __init__(self):
        self.__position = Pose()
        self.__goal = Pose()
        self.__heading = 0
        self.__scan_range = []

        self.__read_parameters()
        self.__init_subscribers()
        self.__init_publishers()

    def __read_parameters(self) -> None:
        self.__odom = rospy.get_param("subscribers/odom")
        self.__scan = rospy.get_param("subscribers/scan")

        self.__cmd_vel = rospy.get_param("publishers/cmd_vel")

    def __init_subscribers(self) -> None:
        rospy.Subscriber(self.__odom, Odometry, \
                         self._odom_callback)
        
        rospy.Subscriber(self.__scan, LaserScan, \
                         self._scan_callback)
        
    def __init_publishers(self) -> None:
        self.__publisher_cmd_vel = rospy.Publisher(self.__cmd_vel, Twist, queue_size=5)

    def _odom_callback(self, odom) -> None:
        self.__position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.__goal.y - self.__position.y, self.__goal.x - self.__position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.__heading = round(heading, 2)

    def _scan_callback(self, scan):

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                self.__scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                self.__scan_range.append(0)
            else:
                self.__scan_range.append(scan.ranges[i])

    def shutdown(self):
        rospy.loginfo("Terminado atividade do TurtleBot")
        self.__publisher_cmd_vel.publish(Twist())
        rospy.sleep(1)
        rospy.signal_shutdown("The End")

    def get_state(self, action):
        current_distance = round(math.hypot(self.__goal.x - self.__position.x, self.__goal.y - self.__position.y),2) 
        
        if 0.13 > min(self.__scan_range) > 0:
            collision = True

        if current_distance < 0.2:
            goal = True