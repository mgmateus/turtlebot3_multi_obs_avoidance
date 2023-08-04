import rospy

import numpy as np

from std_srvs.srv import Empty

from simu.respawnGoal import Respawn 
from turtlebot3 import TurtleBot3

class Environment:
    def __init__(self, services_timeout: float) -> None:
        self.__init_goal = True

        self.__turtle = TurtleBot3()

        self.__read_parameters()
        self.__init_services()
        self.__check_for_services()

    def __read_parameters(self) -> None:
        self.__service_reset_simulation = rospy.get_param("services/reset_simulation")
        self.__service_unpause_physics = rospy.get_param("services/unpause_physics")
        self.__service_pause_physics = rospy.get_param("services/pause_physics")
        
    def __init_services(self) -> None:
        self.__reset_simulation_proxy = rospy.ServiceProxy(self.__service_reset_simulation, Empty)
        self.__unpause_physics_proxy = rospy.ServiceProxy(self.__service_unpause_physics, Empty)
        self.__pause_physics_proxy = rospy.ServiceProxy(self.__service_pause_physics, Empty)

    def __check_for_services(self, services_timeout: float) -> None:
        try:
            rospy.wait_for_service(self.__service_reset_simulation, timeout=services_timeout)
            rospy.wait_for_service(self.__service_unpause_physics, timeout=services_timeout)
            rospy.wait_for_service(self.__service_pause_physics, timeout=services_timeout)

        except rospy.ROSException as ros_exception:
            raise rospy.ROSException from ros_exception

    def _reset_simulation(self) -> bool:
        try:
            self.__reset_simulation_proxy()
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception
        

    def reset(self):
        self._reset_simulation()

        if self.__init_goal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
            
        rospy.loginfo("Target position: x-> %s, y-> %s!!", self.goal_x, self.goal_y)

        self.goal_distance = self.getGoalDistace()
        state, distance, collision, goal = self.getState(data)
        
        self.report_goal_distance(distance, collision, goal)

        return np.asarray(state)