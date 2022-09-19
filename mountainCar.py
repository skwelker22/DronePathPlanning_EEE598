import gym

import matplotlib.pyplot as plt 

env = gym.make('MountainCar-v0',render_mode="rgb_array")



# #Observation and action space
# obs_space = env.observation_space
# action_space = env.action_space

# print("The observation space: {}".format(obs_space))
# print("The action space: {}".format(action_space))

# # reset the environment and see the initial observation
# obs = env.reset()
# print("The initial observation is {}".format(obs))

# # Sample a random action from the entire action space
# random_action = env.action_space.sample()

# # # Take the action and get the new observation space
# new_obs, reward, done, info, extra = env.step(random_action)
# print("The new observation is {}".format(new_obs))

env.reset()
#View environmentin human renderable form:
env_screen = env.render()
env.close()

#Show screenshot
plt.imshow(env_screen)
plt.show()
