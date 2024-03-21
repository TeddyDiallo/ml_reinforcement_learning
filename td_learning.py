import numpy as np
import gymnasium as gym
import random

#details to know before
#We have 6 possible actions, 0 = south, 1 = north, 2 = east, 3 = west, 4 = pick up, 5 = drop off
#the taxi will lose 1 point for every action it takes. They will lose 10 points for inccorect pick up/drop off. A successful drop off will be +20 points
#rewards: each step = -1, incorrect pick up = -10, incorrect drop off = -10, successful drop off = +20
#this program will use TD(0)-learning 

#WORKS!!!
def main():

    # Create taxi environment
    env = gym.make('Taxi-v3', render_mode="rgb_array")

    # Initializing the Q-table and V-table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    qtable = np.zeros((numStates, numActions))
    vtable = np.zeros(numStates)
    
    # Setting a discount rate (gamma), learning rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate = 0.005

    # The model will train for 1000 episodes with 99 steps in each episode
    numEpisodes = 1000
    maxSteps = 99

    for episode in range(numEpisodes):

        # Reset the environment for each episode
        state = [env.reset()]
        done = False

        for step in range(maxSteps):
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                #explores a random action
                action = env.action_space.sample() 
            else:
                # exploits the best action
                action = np.argmax(qtable[state[0][0], :]) 

            # takes in action and observes new_state, reward, if done is T/F, other info that comes from the step() function
            # had to unpack the tuple that was returned from the step() function. (this gave us many issues)
            result = env.step(action)  
            #next state that happens after an action is taken
            next_state = result[0]
            #the reward for reaching this new state
            reward = result[1]
            #checks whether the state has reached the ending or not
            done = result[2]
            #extra info that comes from the step functions
            info = result[3]
            
            print("state:", state[0])
            print("action", action)
            print("next state:", next_state)

            # TD(0) update
            q_current = qtable[state[0][0], action]
            q_next_max = np.max(qtable[next_state, :])
            td_target = reward + discountRate * q_next_max
            td_error = td_target - q_current
            qtable[state[0][0], action] += learningRate * td_error
            
            #update the V-table
            td_target_v = reward + discountRate * vtable[next_state]
            td_error_v = td_target_v - vtable[state[0][0]]
            vtable[state[0][0]] += learningRate * td_error_v

            #update new state
            state[0] = [next_state]
            
            #will end once passenger is dropped of
            if done == True:
                break
            
        # Decay rate using epsilon, so it explores less and exploits more towards the end
        epsilon = np.exp(-decayRate * episode)

    print("TD Learning completed using", numEpisodes, "episodes")
    input("Press Enter to view trained agent")

    # run the trained agent
    state = [env.reset()]
    done = False
    totalRewards = 0

    for step in range(maxSteps):
        print(f"TRAINED TD-LEARNING AGENT")
        print("step:" ,step )
        
        action = np.argmax(qtable[state[0][0], :])
        result = env.step(action)  
        next_state = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]
        
        totalRewards += reward
        state[0] = [next_state]

        env.render()
        print("Score:", totalRewards)

        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()
