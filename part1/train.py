import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
from agent import Policy, Agent

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  
    print('Action space:', env.action_space)      

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = "cpu"
    print(f"Training on '{device}'.")

    # 1. Initialize the policy and agent
    policy = Policy(state_space=state_dim, action_space=action_dim)
    agent = Agent(policy=policy, device=device)

    # --- CHOOSE THE ALGORITHM ---
    # OPTIONS: "REINFORCE", "REINFORCE_BASELINE", "ACTOR_CRITIC"
    ALGORITHM = "ACTOR_CRITIC"
    
    TOTAL_EPISODES = 2000 
    print(f"Starting training! Selected Algorithm: {ALGORITHM}")

    episode_rewards = []

    for episode in range(1, TOTAL_EPISODES + 1):
        state, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            
            # 1. Get action and log probability from the agent's policy
            action, action_log_prob = agent.get_action(state, evaluation=False)
            
            # To ensure compatibility with the environment, we need to convert the action from a PyTorch tensor to a NumPy array
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            action_clean = np.array(action, dtype=np.float32).flatten()

            # 2. Apply the action to the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            ep_reward += reward

            # 3. Store the outcome of the action in the agent's memory
            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            
            state = next_state

        # 4. Episode now over, we need to update the policy based on the collected experience
        
        # We pull out the stored experience from the agent's memory and convert them to tensors
        action_log_probs = torch.stack(agent.action_log_probs, dim=0).to(agent.train_device).squeeze(-1)
        rewards = torch.stack(agent.rewards, dim=0).to(agent.train_device).squeeze(-1)
        states = torch.stack(agent.states, dim=0).to(agent.train_device).squeeze(-1)
        next_states = torch.stack(agent.next_states, dim=0).to(agent.train_device).squeeze(-1)
        dones = torch.Tensor(agent.done).to(agent.train_device)

        # Clear memory after stacking for the next episode
        agent.states, agent.next_states, agent.action_log_probs, agent.rewards, agent.done = [], [], [], [], []

        # --- ALGORITHMIC AND MATHEMATICAL CALCULATIONS ---
        
        if ALGORITHM == "REINFORCE":
            # Task 1: REINFORCE WITHOUT BASELINE
            discounted_rewards = torch.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, rewards.size(-1))):
                running_add = running_add * agent.gamma + rewards[t]
                discounted_rewards[t] = running_add
            
            # Normalize discounted rewards for better training stability
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
            loss = -(action_log_probs * discounted_rewards).sum()
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        elif ALGORITHM == "REINFORCE_BASELINE":
            # Task 2: REINFORCE WITH STABLE BASELINE (b=20)
            discounted_rewards = torch.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, rewards.size(-1))):
                running_add = running_add * agent.gamma + rewards[t]
                discounted_rewards[t] = running_add
            
            # G_t - b=20 formula
            advantages = discounted_rewards - 20
            loss = -(action_log_probs * advantages).sum()
            
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

        elif ALGORITHM == "ACTOR_CRITIC":
            # Task 3: ACTOR-CRITIC
            # We need to compute the state values using the critic network. Since our Policy class only returns the action distribution, we need to manually compute the critic values here.
            x_critic = agent.policy.tanh(agent.policy.fc1_critic(states))
            x_critic = agent.policy.tanh(agent.policy.fc2_critic(x_critic))
            state_values = agent.policy.fc3_critic_value(x_critic).squeeze(-1)
            
            with torch.no_grad():
                x_next_critic = agent.policy.tanh(agent.policy.fc1_critic(next_states))
                x_next_critic = agent.policy.tanh(agent.policy.fc2_critic(x_next_critic))
                next_state_values = agent.policy.fc3_critic_value(x_next_critic).squeeze(-1)
                target_values = rewards + agent.gamma * next_state_values * (1 - dones)
            
            advantages = target_values - state_values
            actor_loss = -(action_log_probs * advantages.detach()).sum()
            critic_loss = F.mse_loss(state_values, target_values, reduction='sum')
            
            total_loss = actor_loss + 0.5 * critic_loss
            
            agent.optimizer.zero_grad()
            total_loss.backward()
            agent.optimizer.step()

        episode_rewards.append(ep_reward)

        # 5. Print training progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} / {TOTAL_EPISODES} -> Average of last 50: {avg_reward:.2f}")

    print("Training completed!")
    torch.save(agent.policy.state_dict(), f"hopper_{ALGORITHM.lower()}_model.pth")
    print("Model successfully saved!")
    env.close()

if __name__ == '__main__':
    main()