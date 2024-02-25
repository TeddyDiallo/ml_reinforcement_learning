import numpy as np
import gymnasium as gym
import random


#details to know before
#We have 6 possible actions, 0 = south, 1 = north, 2 = east, 3 = west, 4 = pick up, 5 = drop off
#the taxi will lose 1 point for every action it takes. They will lose 10 points for inccorect pick up/drop off. A successful drop off will be +20 points
#rewards: each step = -1, incorrect pick up = -10, incorrect drop off = -10, successful drop off = +20
#this program will use SARSA (we modified our q-learning algo.)
def main():

    # create taxi environment
    env = gym.make('Taxi-v3',render_mode="rgb_array")

    #intializing the q-table
    numStates = env.observation_space.n
    numActions = env.action_space.n
    qtable = np.zeros((numStates, numActions))

    #setting a discount rate (gamma), learning rate, and epsilon
    learningRate = 0.9
    discountRate = 0.8
    epsilon = 1.0
    decayRate= 0.005
    
    #the model will train for 1000 episodes with 99 steps in each episode
    numEpisdodes = 1000
    maxSteps = 99
    
    for episode in range(numEpisdodes):

        # reset the environment each episode
        state = [env.reset()]
        done = False
    
        #uses a greedy epsilon policy to decide whether to explore or exploit
        for s in range(maxSteps):
            if random.uniform(0,1) < epsilon:
                # explores a random action
                action = env.action_space.sample()
            else:
                #exploits the best action
                action = np.argmax(qtable[state[0][0],:])
            
            
            # takes in action and observes new_state, reward, if done is T/F, other info that comes from the step() function
            # had to unpack the tuple that was returned from the step() function. (this gave us many issues)
            result = env.step(action)  
            #next state that happens after an action is taken
            newState = result[0]
            #the reward for reaching this new state
            reward = result[1]
            #checks whether the state has reached the ending or not
            done = result[2]
            #extra info that comes from the step functions
            info = result[3]
            
            #this gets us the next best action so we can use it in our SARSA algorithm
            nextAction = np.argmax(qtable[newState,:])
            
            #just to check and make sure the state/action pairs are coming out right. Was used for debugging
            print("State:", state[0][0])
            print("Action:", action)
            # print("Iteration: ", count)
            # count+=1

    
            #SARSA algorithm (just edited the q-learning. changed it from using the max for the next state to simple using the next state/action pair)
            qtable[state[0][0],action] = qtable[state[0][0],action] + learningRate * (reward + discountRate * qtable[newState,nextAction]-qtable[state[0][0],action])
            

            # Update our new state
            print("new state:", newState)
            state[0] = [newState]


            # if done is true, the episode will end. This means that if teh passenger is dropped off, it will become true and end.
            if done == True:
                break

        #this will decrease epsilon (used for our greedy epsilon policy
        #did not fully understand this code, but I read about it online and found that it decreases epsilon exponentially to ensure our agent explores less and less
        epsilon = np.exp(-decayRate*episode)

    #concludes the training
    print("SARSA training completed using", numEpisdodes, "episodes")
    input("Press Enter to view trained agent")

    
    #we can now watch the trained agent
    state = [env.reset()]
    done = False
    totalRewards = 0

    #runs for as many steps as it takes for the taxi to pick up the passenger and drop them off correctly
    for s in range(maxSteps):

        print(f"TRAINED SARSA AGENT")
        print("Step: ", s)

        #the best actions
        action = np.argmax(qtable[state[0][0],:])
        
        #same steps as above
        result = env.step(action)  
        newState = result[0]
        reward = result[1]
        done = result[2]
        info = result[3]

        #increase/decrease the cummilative rewards
        totalRewards += reward
        #increase step count
        s += 1
        
        env.render()
        print("score: ", totalRewards)
        #update new state
        state[0] = [newState]

        #the end (the taxi has had a successful drop off. We know this because our second to last step is always a negative value, and the last step is that value + 20 (successful drop off))
        if done == True:
            break

    env.close()

if __name__ == "__main__":
    main()