from operator import truediv
import numpy as np
import gym
import random
import time
from IPython.display import clear_output

#Get the object_handle for the FrozenLake environment:
#env = gym.make("FrozenLake-v1",map_name="8x8", is_slippery=False)
env = gym.make("FrozenLake-v1",desc=["SFHF", "HFFH", "HHFH", "HFFG"], is_slippery=False)

#Query for the states and actions
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

#print(action_space_size)
#print(state_space_size)

#Create the Q-table:
q_table = np.zeros((state_space_size,action_space_size))
print(q_table)

#Global settings/Hyperparameters for the Reinforcement Learning game:
num_episodes = 10000 #numer of episodes
max_steps_per_episode = 100 #number of actions/steps per episode

learning_rate = 0.1 #we prioritze past data more
discount_rate = 0.99 # usual discount rate

exploration_rate = 1 # This is the starting rate, which we will decay
max_exploration_rate = 1 # upper ceiling 
min_exploration_rate = 0.01 # lower ceiling
exploration_recay_rate = 0.001 # rate of decay.

#Begin the Q-learning algorithm:
rewards_all_episodes = []

for episode in range(num_episodes):
    state= env.reset()
   

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        exploration_rate_threshold = random.uniform(0,1)

        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
            

        new_state, reward, done, info = env.step(action)
      
        #Update Q-table for Q(s,a)
        q_table[state,action] = q_table[state,action]*(1-learning_rate) + learning_rate*(reward + discount_rate * np.max(q_table[new_state,:]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

#exploration rate decay
exploration_rate = min_exploration_rate + (max_exploration_rate-min_exploration_rate)*np.exp(-exploration_recay_rate*episode)
rewards_all_episodes.append(rewards_current_episode)

#Calculate and print the average reward per 1000 episodes

reward_per_thousand_episodes = np.array_split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("****Average REwards for 1000 episodes")
for r in reward_per_thousand_episodes:
    print(count,":",str(sum(r/1000)))
    count += 1000 

#Print updated Q-table
print("\n\n***************Q-table*************\n")
print(q_table)


###Run trained AI to see how it performs:
for episode in range(3):
    state = env.reset()
    done = False
    print("Episde",episode+1,"\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render(mode = "human")
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state,reward,done,info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render(mode = "human")
            if reward == 1:
                print("** You reached goal")
                time.sleep(3)

            else:
                env.render(mode = "human")
                print("You fell in hole")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state
    env.close()


