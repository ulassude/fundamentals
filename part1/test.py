import gymnasium as gym
import torch
import numpy as np
from agent import Policy #

env = gym.make('Hopper-v4', render_mode='human')
policy = Policy(state_space=11, action_space=3)

policy.load_state_dict(torch.load("hopper_reinforce_model.pth"))
#policy.load_state_dict(torch.load("hopper_reinforce_baseline_model.pth"))
#policy.load_state_dict(torch.load("hopper_actor_critic_model.pth"))

policy.eval()

state, info = env.reset()
done = False

while not done:
    state_clean = np.array(state, dtype=np.float32)
    normal_dist = policy(torch.from_numpy(state_clean))
    action = normal_dist.mean.detach().numpy()
    
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated