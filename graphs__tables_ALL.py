import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def run_q_learning(env, qtable, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    episode_rewards = []
    for episode in range(num_episodes):
        state = [env.reset()]
        total_rewards = 0
        for _ in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  
            else:
                action = np.argmax(qtable[state[0][0], :])  
            
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            done = result[2]
            info = result[3]
            
            qtable[state[0][0], action] += learning_rate * (
                    reward + discount_rate * np.max(qtable[next_state, :]) - qtable[state[0][0], action])
            
            state[0] = [next_state]
            total_rewards += reward
            if done:
                break
        
        epsilon = np.exp(-decay_rate * episode)
        episode_rewards.append(total_rewards)
    
    return episode_rewards

def run_sarsa(env, qtable, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    episode_rewards = []
    for episode in range(num_episodes):
        state = [env.reset()]
        total_rewards = 0
        for _ in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state[0][0], :])
            
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            done = result[2]
            info = result[3]
            
            next_action = np.argmax(qtable[next_state, :])
            qtable[state[0][0], action] += learning_rate * (reward + discount_rate * qtable[next_state, next_action] - qtable[state[0][0], action])
            
            state[0] = [next_state]
            total_rewards += reward
            if done:
                break
        
        epsilon = np.exp(-decay_rate * episode)
        episode_rewards.append(total_rewards)
    
    return episode_rewards

def run_td_learning(env, qtable, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    episode_rewards = []
    for episode in range(num_episodes):
        state = [env.reset()]
        total_rewards = 0
        for _ in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state[0][0], :])
            
            result = env.step(action)
            next_state = result[0]
            reward = result[1]
            done = result[2]
            info = result[3]
            
            q_current = qtable[state[0][0], action]
            q_next_max = np.max(qtable[next_state, :])
            td_target = reward + discount_rate * q_next_max
            td_error = td_target - q_current
            qtable[state[0][0], action] += learning_rate * td_error
            
            state[0] = [next_state]
            total_rewards += reward
            if done:
                break
        
        epsilon = np.exp(-decay_rate * episode)
        episode_rewards.append(total_rewards)
    
    return episode_rewards

def compute_statistics(rewards):
    average_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return average_reward, std_reward
import matplotlib.pyplot as plt

def plot_rewards(q_learning_rewards, sarsa_rewards, td_learning_rewards, num_runs):
    q_learning_rewards /= num_runs
    sarsa_rewards /= num_runs
    td_learning_rewards /= num_runs
        
    plt.figure(figsize=(10, 6))
    plt.plot(q_learning_rewards, label='Q-Learning')
    plt.plot(sarsa_rewards, label='SARSA')
    plt.plot(td_learning_rewards, label='TD-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning vs SARSA vs TD-Learning Performance')
    plt.legend()
    plt.show()
    
def display_table(q_learning_avg, q_learning_std, sarsa_avg, sarsa_std, td_learning_avg, td_learning_std):
    labels = ['Algorithm', 'Mean', 'STD. DEV.']
    
    # Round total rewards
    q_learning_avg = round(q_learning_avg, 2)
    sarsa_avg = round(sarsa_avg, 2)
    td_learning_avg = round(td_learning_avg, 2)
    q_learning_std = round(q_learning_std, 2)
    sarsa_std = round(sarsa_std, 2)
    td_learning_std = round(td_learning_std, 2)
    data = [
        ['Q-Learning', q_learning_avg, q_learning_std],
        ['SARSA', sarsa_avg, sarsa_std],
        ['TD-Learning', td_learning_avg, td_learning_std]
    ]
    
    fig, ax = plt.subplots(figsize=(4, 3))  
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=labels, cellLoc='center', loc='center')

    # Formatting
    table.auto_set_font_size(False)
    table.set_fontsize(10) 
    table.scale(1.2, 1.2)  
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontsize=12, weight='bold', fontfamily='Times New Roman')  
        else:
            cell.set_text_props(fontsize=10, fontfamily='Times New Roman')  
    plt.title('Algorithm Performance', fontsize=14, fontweight='bold', fontfamily='Times New Roman')  
    plt.show()


def main():
    env = gym.make('Taxi-v3', render_mode="rgb_array")
    
    num_runs = 10
    num_episodes = 1000
    max_steps = 99
    
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005
    
    q_learning_rewards = np.zeros(num_episodes)
    sarsa_rewards = np.zeros(num_episodes)
    td_learning_rewards = np.zeros(num_episodes)
    
    for run in range(num_runs):
        qtable_q_learning = np.zeros((env.observation_space.n, env.action_space.n))
        qtable_sarsa = np.zeros((env.observation_space.n, env.action_space.n))
        qtable_td_learning = np.zeros((env.observation_space.n, env.action_space.n))

        
        episode_rewards_q_learning = run_q_learning(env, qtable_q_learning, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes)
        episode_rewards_sarsa = run_sarsa(env, qtable_sarsa, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes)
        episode_rewards_td_learning = run_td_learning(env, qtable_td_learning, learning_rate, discount_rate, epsilon, decay_rate, max_steps, num_episodes)

        q_learning_rewards += np.array(episode_rewards_q_learning)
        sarsa_rewards += np.array(episode_rewards_sarsa)
        td_learning_rewards += np.array(episode_rewards_td_learning)
    
    #THIS WILL PLOT THE GRAPH
    # plot_rewards(q_learning_rewards, sarsa_rewards, td_learning_rewards, num_runs)

    # Compute statistics 
    q_learning_avg, q_learning_std = compute_statistics(q_learning_rewards)
    sarsa_avg, sarsa_std = compute_statistics(sarsa_rewards)
    td_learning_avg, td_learning_std = compute_statistics(td_learning_rewards)
    
    
    # THIS WILL CREATE GRAPH
    # YOU NEED TO COMMENT THIS OUT TO SEE THE GRAPH!!! can only see one at a time!!
    display_table(q_learning_avg, q_learning_std, sarsa_avg, sarsa_std, td_learning_avg, td_learning_std)
    
    
if __name__ == "__main__":
    main()
