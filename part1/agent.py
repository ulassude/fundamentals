import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        # Critic maps state_space to a single scalar state-value V(s)
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                # Normal initialization with small std to prevent early saturation
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward in the critic network
        # Forward pass through the critic network to estimate V(s)
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic)
        
        # Return both the action distribution and estimated state value
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        # Using a stable learning rate of 3e-4 to avoid policy collapse
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

    def update_policy(self, algorithm="REINFORCE_BASELINE", b=20):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        # TASK 2: REINFORCE with Constant Baseline
        #   - compute discounted returns
        #   - compute policy gradient loss function given actions and returns
        #   - compute gradients and step the optimizer
        #
        if algorithm == "REINFORCE_BASELINE":
            # 1. Compute discounted returns G_t using the helper function
            returns = discount_rewards(rewards, self.gamma)
            
            # 2. Calculate the advantage using the stable constant baseline (b)
            advantages = returns - b
            
            # 3. Compute policy gradient loss (negative for gradient ascent)
            policy_loss = -(action_log_probs * advantages).sum()
            
            # 4. Backpropagation and optimization step
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        #
        # TASK 3: Actor-Critic Algorithm
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #
        elif algorithm == "ACTOR_CRITIC":
            # 1. Get current state value estimates V(s) from critic
            _, state_values = self.policy(states)
            state_values = state_values.squeeze(-1)
            
            # 2. Compute bootstrapped TD targets: r + gamma * V(s') * (1 - done)
            with torch.no_grad():
                _, next_state_values = self.policy(next_states)
                next_state_values = next_state_values.squeeze(-1)
                target_values = rewards + self.gamma * next_state_values * (1 - done)
            
            # 3. Compute TD Advantage: Target - V(s)
            advantages = target_values - state_values
            
            # 4. Calculate Actor loss (Policy Gradient) and Critic loss (MSE)
            actor_loss = -(action_log_probs * advantages.detach()).sum()
            critic_loss = F.mse_loss(state_values, target_values, reduction='sum')
            
            # Total multi-task loss with standard Critic weight scaling (0.5)
            total_loss = actor_loss + 0.5 * critic_loss
            
            # 5. Joint optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return         

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        # Unpack the tuple since policy forward now returns (normal_dist, state_value)
        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)