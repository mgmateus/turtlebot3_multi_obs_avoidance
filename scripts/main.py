#!/usr/bin/env python3

import rospy
import time
import copy

import numpy as np

from modules.ou_noise import OUNoise
from environment import Environment
from ddpg import DDPGagent


if __name__ == "__main__":
    rospy.init_node("turtlebot3_multi_obs_avoidance")

    SPACE_STATE_DIM= rospy.get_param("params/state_dimension")
    SPACE_ACTION_DIM= rospy.get_param("params/action_dimension")
    MAX_STEPS= rospy.get_param("params/max_steps")
    MAX_EPS= rospy.get_param("params/num_episodes")
    BUFFER_SIZE= rospy.get_param("params/num_episodes")
    BATCH_SIZE= rospy.get_param("params/batch_size")

    ALPHA = rospy.get_param("params/alpha_reward")
    BETHA = rospy.get_param("params/betha_reward")

    STOPED = rospy.get_param("params/stoped_episode")
    EPS= rospy.get_param("params/stoped_episode")

    is_training = rospy.get_param('~training')


    env = Environment(SPACE_STATE_DIM, SPACE_ACTION_DIM, MAX_STEPS)
    env.alpha = ALPHA
    env.betha = BETHA
    agent = DDPGagent(SPACE_STATE_DIM, SPACE_ACTION_DIM, buffer_size=BUFFER_SIZE)
    noise = OUNoise(SPACE_ACTION_DIM)

    if STOPED > 0:
        
        agent.load_models(EPS)
        rospy.logwarn("Load Model %s", EPS)

    for eps in range(EPS, MAX_EPS):
        rospy.logwarn("Current Episode: %s", eps)
        rospy.set_param("params/stoped_episode", eps)

        done = False
        state = env.reset()
        noise.reset()
        rewards_current_episode = 0.0
        
        for step in range(MAX_STEPS):
            state = np.float32(state)
            action = agent.get_action(state)
            
            N = copy.deepcopy(noise.get_noise(t=step))        
            action[0] = np.clip(action[0] + (N[0]*.75), -1, 1)
            action[1] = np.clip(action[1] + (N[1]*.75), -1, 1)
                          
            new_state, reward, done = env.step(action) 
            rewards_current_episode += reward
            new_state = np.float32(new_state)

            if rospy.get_param("params/max_reward") < reward:
                rospy.set_param("params/max_reward", float(reward))

            if not eps%10 == 0 or not len(agent.memory) >= SPACE_ACTION_DIM*BATCH_SIZE:
                if done:
                    for _ in range(3):
                        agent.memory.push(state, action, reward, new_state, done)
                else:
                    agent.memory.push(state, action, reward, new_state, done)

            if is_training and len(agent.memory) > SPACE_ACTION_DIM*BATCH_SIZE and not eps%10 == 0:
                #rospy.logwarn("--------- UPDATE ----------")
                agent.update(BATCH_SIZE)
            
            rospy.logwarn(f"Reward: {reward}")
            state = copy.deepcopy(new_state)   
            
            agent.save_state(eps, step, reward, state)

            if done or step == MAX_STEPS-1:
                rospy.logwarn("Reward per ep: %s", str(rewards_current_episode))
                rospy.logwarn("Break step: %s", str(step))
                rospy.logwarn("Sigma: %s", str(noise.sigma))
            
                break
        if is_training:
            agent.save_models(eps)
            agent.save_rewards(eps, rewards_current_episode)