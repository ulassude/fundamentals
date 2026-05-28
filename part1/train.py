"""Sample script for training a control policy on the Hopper environment

    Here you will implement the training loop for REINFORCE and Actor-Critic
"""
import gymnasium as gym
import torch
import numpy as np
from agent import Policy, Agent 

def main():
    env = gym.make('Hopper-v4')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space

    # Dynamically extract state and action spaces dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Target device configuration (CPU is highly recommended here to avoid driver-specific CUDA mismatches)
    device = "cpu"
    print(f"Training execution device set to: '{device}'.")

    # Initialize the Policy network and Agent using the exact template interfaces
    policy = Policy(state_space=state_dim, action_space=action_dim)
    agent = Agent(policy=policy, device=device)

    # --- CHOOSE THE ALGORITHM TO EXECUTE ---
    # OPTIONS: "REINFORCE_BASELINE" (TASK 2) or "ACTOR_CRITIC" (TASK 3)
    ALGORITHM = "ACTOR_CRITIC" 
    TOTAL_EPISODES = 2000 
    
    print(f"Starting training loop! Selected Mode: {ALGORITHM}")

    episode_rewards = []

    for episode in range(1, TOTAL_EPISODES + 1):
        state, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            # Cast the environment state to float32 numpy array to bypass internal type locks
            state_clean = np.array(state, dtype=np.float32)
            
            # Step 1: Request action selection and its log probability from the agent
            action, action_log_prob = agent.get_action(state_clean, evaluation=False)
            
            # Safe conversion: isolate the action tensor to a pure NumPy float32 array for MuJoCo
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
            action_clean = np.array(action, dtype=np.float32).flatten()

            # Step 2: Execute the clean action within the Gymnasium environment
            next_state, reward, terminated, truncated, info = env.step(action_clean)
            
            done = terminated or truncated
            ep_reward += reward

            # Step 3: Record transaction step into the agent's memory lists
            agent.store_outcome(state, next_state, action_log_prob, reward, done)
            
            # Update local state reference
            state = next_state

        # Append total accumulated episodic reward to tracking array
        episode_rewards.append(ep_reward)

        # Step 4: Episode terminated. Trigger the agent's encapsulated update function.
        # All policy gradient, returns and advantage calculations execute inside agent.py
        agent.update_policy(algorithm=ALGORITHM, b=20)
        
        # Step 5: Print tracking diagnostics progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} / {TOTAL_EPISODES} -> Average of last 50 episodes: {avg_reward:.2f}")

    print("Training phase fully completed!")
    
    # Save model weights named dynamically based on the trained algorithm
    model_save_path = f"hopper_{ALGORITHM.lower()}_model.pth"
    torch.save(agent.policy.state_dict(), model_save_path)
    print(f"Model checkpoint successfully saved to {model_save_path}!")
    
    env.close()

if __name__ == '__main__':
    main()