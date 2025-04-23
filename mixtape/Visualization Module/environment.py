# environment.py - Environment setup and episode running

import os
import gym
import crafter
import numpy as np
import pandas as pd
import cv2
from dreamer_policy import DreamerPolicy

def create_environment(output_dir):
    """Create and configure the Crafter environment"""
    env = gym.make('CrafterReward-v1')
    return env

def run_episode(env, policy_fn, max_steps=1000, record_video=True, output_dir='./logs_dreamer'):
    """Run a single episode using the provided policy function"""
    # Create directories if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize recording
    if record_video:
        video_path = os.path.join(output_dir, 'episode.mp4')
        frame_size = (256, 256)  # Default Crafter size
        fps = 30
        video_writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    # Initialize data collection
    observations = []
    actions = []
    rewards = []
    dones = []
    cumulative_reward = 0
    reward_components = {}
    
    # Start episode
    obs = env.reset()
    
    # Run until episode ends or max steps reached
    for step in range(max_steps):
        # Get action from policy
        action = policy_fn(obs)
        
        # Take action in environment
        next_obs, reward, done, info = env.step(action)
        
        # Record video frame if enabled
        if record_video:
            frame = obs.copy()  # Make a copy of the observation
            video_writer.write(frame)
        
        # Store data
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        # Update cumulative reward
        cumulative_reward += reward
        
        # Extract reward components from info
        for key, value in info.items():
            if key not in reward_components:
                reward_components[key] = []
            reward_components[key].append(value)
        
        # Update current observation
        obs = next_obs
        
        # Check if episode is done
        if done:
            break
    
    # Close video writer if recording
    if record_video:
        video_writer.release()
    
    # Create results dataframe
    results = {
        'time_step': list(range(len(actions))),
        'action': actions,
        'reward': rewards,
        'cumulative_reward': np.cumsum(rewards).tolist()
    }
    
    # Add reward components to results
    for key, values in reward_components.items():
        results[key] = values
    
    # Save as CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'episode_data.csv')
    results_df.to_csv(csv_path, index=False)
    
    print(f"Episode completed with total reward: {cumulative_reward:.2f}")
    print(f"Results saved to {csv_path}")
    if record_video:
        print(f"Video saved to {video_path}")
    
    return results

def dreamer_policy(env, agent, obs):
    """Policy function that uses a DreamerV2 agent."""
    action = agent.policy(obs)
    return action

def run_with_dreamer(output_dir='./logs_dreamer'):
    """Run episode with DreamerV2 agent"""
    
    # Create environment
    env = create_environment(output_dir)
    
    # Create DreamerV2 agent
    agent = DreamerPolicy(
        env,
        training=False,
        checkpoint_dir='./dreamer_checkpoints',
        load_checkpoint=True
    )
    
    # Create policy function that uses the agent
    def policy_func(obs):
        return agent(obs)
    
    # Run episode with visualization
    results = run_episode(env, policy_func)
    
    return results
