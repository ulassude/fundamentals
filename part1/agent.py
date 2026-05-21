import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=11, action_dim=3, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Mean and Std for action distribution
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        # Standard deviation cannot be negative, so we'll keep log_std and then take exp()
        self.action_log_std = nn.Parameter(torch.full((1, action_dim), -1.0))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.action_mean(x)) # Hopper actions are typically in the range [-1, 1], so we can use tanh to bound the mean
        std = torch.exp(self.action_log_std)
        
        return mean, std

class REINFORCEAgent:
    def __init__(self, state_dim=11, action_dim=3):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        
        # Memory lists to store log probabilities and rewards for each episode
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean, std = self.policy(state)
        
        dist = Normal(mean, std)
        action = dist.sample()
        
        # LETS DO THE SQUEEZE TRICK TO AVOID LOG_PROB SHAPE ISSUES
        # First we sample the action, then we squeeze it to remove any extra dimensions, and finally we calculate the log probability of that action
        action_squeezed = action.squeeze(0)
        log_prob = dist.log_prob(action_squeezed).sum(dim=-1)
        self.log_probs.append(log_prob)
        
        return action_squeezed.detach().numpy()
    
    def update(self, gamma=0.99):
            discounted_rewards = []
            G = 0
            
            # Calculate discounted rewards in reverse order (from the end of the episode to the beginning)
            for r in reversed(self.rewards):
                G = r + gamma * G
                discounted_rewards.insert(0, G)
                
            discounted_rewards = torch.FloatTensor(discounted_rewards)
            
            # Normalize discounted rewards for better training stability
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
            
            policy_loss = []
            for log_prob, Gt in zip(self.log_probs, discounted_rewards):
                policy_loss.append(-log_prob * Gt)
                
            # Gradient update
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            
            # Clear memory
            self.log_probs = []
            self.rewards = []
            
    def update2(self, gamma=0.99, b=20): # b=20 baseline
        discounted_rewards = []
        G = 0
        
        # Calculate discounted rewards in reverse order (from the end of the episode to the beginning)
        for r in reversed(self.rewards):
            G = r + gamma * G
            discounted_rewards.insert(0, G)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # Note: We are not normalizing the rewards here because we are using a baseline (b=20) to reduce variance, and the advantage calculation will already help with stability. Normalizing after subtracting the baseline could potentially distort the advantage estimates, so we will skip normalization in this case.
        
        policy_loss = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            # Calculating (G_t - b)
            advantage = Gt - b 
            policy_loss.append(-log_prob * advantage)
            
        # Gradient update
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs = []
        self.rewards = []
            