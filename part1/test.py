import gymnasium as gym
import torch
import time
from agent import REINFORCEAgent

# 1. Open the environment with rendering enabled to visualize the agent's performance
env = gym.make("Hopper-v4", render_mode="human")

# 2. Create an agent and load the trained model
agent = REINFORCEAgent()
agent.policy.load_state_dict(torch.load("hopper_reinforce_model.pth"))
agent.policy.eval() # Modeli test (evaluation) moduna alıyoruz

# 3. Watch the agent perform in the environment
state, info = env.reset()
done = False

while not done:
    # We force the agent to select the mean action (deterministic) for visualization purposes
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        mean, _ = agent.policy(state_tensor)
        action = mean.squeeze(0).numpy()
        
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    
    # Sleep between steps to slow down the rendering (optional, adjust as needed)
    time.sleep(0.02) 
    
    done = terminated or truncated

env.close()