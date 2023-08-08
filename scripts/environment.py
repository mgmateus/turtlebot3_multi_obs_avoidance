import rospy
import random
import time
import os
import numpy as np

from std_srvs.srv import Empty
from gazebo_msgs.srv import (
    SpawnModel, 
    DeleteModel
)
from gazebo_msgs.msg import (
    ModelState,
    ModelStates
)
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose

from turtlebot3 import TurtleBot3

class Respawn():
    def __init__(self, services_timeout: float):
        self.__model = None

        self.__model_name = 'goal'
        self.__stage = 4

        self.__goal_position = Pose()
        self.__init_goal_x = 0.6
        self.__init_goal_y = 0.0
        self.__goal_position.position.x = self.__init_goal_x
        self.__goal_position.position.y = self.__init_goal_y
        self.__last_goal_x = self.__init_goal_x
        self.__last_goal_y = self.__init_goal_y

        self.__obstacle_1 = 0.6, 0.6
        self.__obstacle_2 = 0.6, -0.6
        self.__obstacle_3 = -0.6, 0.6
        self.__obstacle_4 = -0.6, -0.6
        
        self.__index = 0
        self.__last_index = 0

        self.__check_model = False

        
        self.__read_files()
        self.__read_parameters()
        self.__init_services()
        self.__init_subscribers()
        self.__check_for_services(services_timeout)

    def __read_files(self):
        model_path = os.path.dirname(os.path.realpath(__file__)).replace('turtlebot3_multi_obs_avoidance/scripts',
                                                                         'turtlebot3_multi_obs_avoidance/world/model.sdf')

        with open(model_path) as file:
            self.__model = file.read()

    def __read_parameters(self) -> None:
        self.__service_name_spawn_sdf_model = rospy.get_param("simulation/services/spawn_sdf_model")
        self.__service_name_delete_model = rospy.get_param("simulation/services/delete_model")

        self.__subscriber_name_model_states = rospy.get_param("simulation/subscribers/model_states")

    def __init_services(self) -> None:
        self.__spawn_sdf_model_proxy = rospy.ServiceProxy(self.__service_name_spawn_sdf_model, SpawnModel)
        self.__delete_model_proxy = rospy.ServiceProxy(self.__service_name_delete_model, DeleteModel)

    def __init_subscribers(self) -> None:
        rospy.Subscriber(self.__subscriber_name_model_states, ModelStates, self.__model_callback)

    def __check_for_services(self, services_timeout: float) -> None:
        try:
            rospy.wait_for_service(self.__service_name_spawn_sdf_model, timeout=services_timeout)
            rospy.wait_for_service(self.__service_name_delete_model, timeout=services_timeout)
        except rospy.ROSException as ros_exception:
            raise rospy.ROSException from ros_exception
        
    def __model_callback(self, model):
        self.__check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.__check_model = True

    def _respawn_model(self) -> bool:
        try:
            self.__spawn_sdf_model_proxy(self.__model_name, self.__model, 'robotos_name_space', self.__goal_position, "world")
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception
        
    def _delete_model(self) -> bool:
        try:
            if self.__check_model:
                self.__delete_model_proxy(self.__model_name)
                rospy.loginfo("Goal position : %.1f, %.1f", self.__goal_position.position.x,
                              self.__goal_position.position.y)
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception
        

    def get_position(self, position_check=False, delete=False):
        if delete:
            self._delete_model()

        if self.__stage != 4:
            while position_check:
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.__obstacle_1[0]) <= 0.4 and abs(goal_y - self.__obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.__obstacle_2[0]) <= 0.4 and abs(goal_y - self.__obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.__obstacle_3[0]) <= 0.4 and abs(goal_y - self.__obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.__obstacle_4[0]) <= 0.4 and abs(goal_y - self.__obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.__last_goal_x) < 1 and abs(goal_y - self.__last_goal_y) < 1:
                    position_check = True

                self.__goal_position.position.x = goal_x
                self.__goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x_list = [1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [-0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.__index = random.randrange(0, 12)
                print(self.__index, self.__last_index)
                if self.__last_index == self.__index:
                    position_check = True
                else:
                    self.__last_index = self.__index
                    position_check = False

                self.__goal_position.position.x = goal_x_list[self.__index]
                self.__goal_position.position.y = goal_y_list[self.__index]

        time.sleep(0.5)
        self._respawn_model()

        self.__last_goal_x = self.__goal_position.position.x
        self.__last_goal_y = self.__goal_position.position.y

        return self.__goal_position.position.x, self.__goal_position.position.y

class Environment(Respawn):
    def __init__(self, state_dim, action_dim, max_steps, services_timeout: float = 60) -> None:
        Respawn.__init__(self, services_timeout)

        self.__turtle = TurtleBot3()
        self.__observation_space = np.zeros(shape=(state_dim,))
        self.__action_dim = action_dim

        self.__n_steps = 0
        self.__max_steps = max_steps

        self.__collision_numbers = 0
        self.__goal_numbers = 10
        self.__init_goal = True

        self.__alpha = 1.85
        self.__betha = 185.71

        self.repulse = lambda d, d0, betha : 0.5*betha*((1/d)-(1/d0))

        self.__read_parameters()
        self.__init_services()
        self.__check_for_services(services_timeout)

    def __read_parameters(self) -> None:
        self.__service_name_reset_simulation = rospy.get_param("simulation/services/reset_simulation")
        self.__service_name_reset_world = rospy.get_param("simulation/services/reset_world")
        self.__service_name_set_model_state = rospy.get_param('simulation/services/set_model_state')
        self.__service_name_delete_model = rospy.get_param("simulation/services/delete_model")
        self.__service_name_pause_physics = rospy.get_param("simulation/services/pause_physics")
        self.__service_name_unpause_physics = rospy.get_param("simulation/services/unpause_physics")
        
    def __init_services(self) -> None:
        self.__reset_simulation_proxy = rospy.ServiceProxy(self.__service_name_reset_simulation, Empty)
        self.__pause_physics_proxy = rospy.ServiceProxy(self.__service_name_pause_physics, Empty)
        self.__unpause_physics_proxy = rospy.ServiceProxy(self.__service_name_unpause_physics, Empty)
        self.__reset_world_proxy = rospy.ServiceProxy(self.__service_name_reset_world, Empty)
        self.__set_model_state_proxy = rospy.ServiceProxy(self.__service_name_set_model_state, SetModelState)
        self.__delete_model_proxy = rospy.ServiceProxy(self.__service_name_delete_model, DeleteModel)

    def __check_for_services(self, services_timeout: float) -> None:
        try:
            rospy.wait_for_service(self.__service_name_reset_simulation, timeout=services_timeout)
            rospy.wait_for_service(self.__service_name_reset_world, timeout=services_timeout)
            rospy.wait_for_service(self.__service_name_set_model_state, timeout=services_timeout)
            rospy.wait_for_service(self.__service_name_delete_model, timeout=services_timeout)
        except rospy.ROSException as ros_exception:
            raise rospy.ROSException from ros_exception

    def _reset_simulation(self) -> bool:
        try:
            self.__pause_physics_proxy()
            self.__unpause_physics_proxy()
            self.__reset_simulation_proxy()
            
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception
        
    def _reset_world(self) -> bool:
        try:
            self.__pause_physics_proxy()
            self.__reset_world_proxy()
            self.__unpause_physics_proxy()
            
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception
        
    def _set_model_state(self) -> bool:
        try:

            self.__pause_physics_proxy()
            
            state_msg = ModelState()
            state_msg.model_name = 'turtlebot3_burger'
            state_msg.pose.position.x = -0.7
            state_msg.pose.position.y = 0
            state_msg.pose.position.z = 0
            state_msg.pose.orientation.x = 0
            state_msg.pose.orientation.y = 0
            state_msg.pose.orientation.z = 0
            state_msg.pose.orientation.w = 0

            self.__set_model_state_proxy(state_msg)

            self.__unpause_physics_proxy()
            return True
        except rospy.ServiceException as service_exception:
            raise rospy.ServiceException from service_exception

    def _gravitational_potential_field(self):
        #beta, alpha
        return 0.5 * self.__alpha * self.__turtle.euclidian_distance_to_goal()
    
    def _repulsive_potential_field(self):
        collision_warn = self.__turtle.collision_warn()
        left = self.repulse(self.__turtle.current_scan['left'], self.__turtle.past_scan['left'], self.__betha) \
                if collision_warn[0] else 0
        
        forward = self.repulse(self.__turtle.current_scan['forward'], self.__turtle.past_scan['forward'], self.__betha) \
                if collision_warn[1] else 0
        
        right = self.repulse(self.__turtle.current_scan['right'], self.__turtle.past_scan['right'], self.__betha) \
                if collision_warn[2] else 0
        
        backward = self.repulse(self.__turtle.current_scan['backward'], self.__turtle.past_scan['backward'], self.__betha) \
                if collision_warn[3] else 0
        
        return left + forward + right + backward

    def reset(self):
        self._set_model_state()
        
        self.__n_steps = 0
        _ = self.__turtle.get_state(np.zeros(shape=(self.__action_dim,)))
        
        return self.__observation_space
     
    def set_reward(self, done):
        inital_distance = self.__turtle.initial_euclidian_distance_to_goal()
        if self.__init_goal:
            self.__turtle.goal = self.get_position()
            self.__init_goal = False
        if done:
            self.__turtle.stop()
            if inital_distance >= 2.0: #alvo mais distante 3.6
                self.__goal_numbers -= 1
                self.__turtle.goal = self.get_position(position_check=True, delete=True)

            rospy.loginfo("**********")
            rospy.loginfo("WELL DONE!!")
            rospy.loginfo("**********")

        if self.__turtle.is_collision():
            self.__collision_numbers += 1
            self._set_model_state()
            rospy.loginfo("**********")
            rospy.loginfo("COLLISION!!")
            rospy.loginfo("**********")

        return self._gravitational_potential_field() + self._repulsive_potential_field()
        
    def step(self, action):
        self.__observation_space, done = self.__turtle.get_state(action)
        reward = self.set_reward(done)
        self.__n_steps += 1
        
        if self.__n_steps < self.__max_steps:
            return np.asarray(self.__observation_space), reward, done
        else:
            return np.asarray(self.__observation_space), 0.0, True