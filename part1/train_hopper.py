import gymnasium as gym
import torch
from agent import REINFORCEAgent

env = gym.make("Hopper-v4")
agent = REINFORCEAgent()

for episode in range(2000): # 2000 episodes for training
    state, info = env.reset()
    done = False
    ep_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        agent.rewards.append(reward)
        state = next_state
        ep_reward += reward
        done = terminated or truncated
        
    # Episodes finished - update policy
    agent.update()
    
    if episode % 10 == 0:
        print(f"Episode {episode} - Total Reward: {ep_reward:.2f}")

torch.save(agent.policy.state_dict(), "hopper_reinforce_model.pth")