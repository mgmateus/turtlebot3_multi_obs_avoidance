import rospy

import numpy as np

from std_srvs.srv import Empty

from simu.respawnGoal import Respawn 
from turtlebot3 import TurtleBot3

class Environment:
    def __init__(self, state_dim, action_dim, max_steps, services_timeout: float) -> None:
        self.__turtle = TurtleBot3()
        self.__observation_space = np.zeros(shape=(state_dim,))
        self.__action_dim = action_dim

        self.__n_steps = 0
        self.__max_steps = max_steps
        self.__past_distance = None

        self.__targets = [(2.8, 3.0), (2.0, -0.8), (0.0, 5.0), (-2.0, 0.0), (-2.0, 0.0), (4.0, 5.0), (-4.0, 6.0), (-2.0, 6.0)]
        self.__target_reset = 0


        self.__collision_numbers = 0
        self.__init_goal = True


        self.__read_parameters()
        self.__init_services()
        self.__check_for_services(services_timeout)

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
        
    def _report_goal_distance(self, distance, collision, goal):
        if collision:
            rospy.loginfo("Collision!!")
            self.__collision_numbers += 1
        if goal:
            rospy.loginfo("Goal!!")
            self.__turtle.stop()
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            distance = self.goal_distance
            rospy.loginfo("Target position: x-> %s, y-> %s!!", self.goal_x, self.goal_y)
            self.goal_numbers -= 1
        
        rospy.loginfo("Number of targets %s / distance to curent goal %s / collission number %s", self.goal_numbers, distance, self.collision_numbers)

    def reset(self):
        self._reset_simulation()

        if self.__init_goal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False
            
        self.__n_steps = 0
        _ = self.__robot.get_state(np.zeros(shape=(self.__action_dim,)))
        
        return self.__observation_space

        return np.asarray(state)