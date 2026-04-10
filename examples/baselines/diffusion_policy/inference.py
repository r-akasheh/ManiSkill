import gymnasium as gym
import torch
import numpy as np
from gymnasium import spaces

from mani_skill.envs.sapien_env import BaseEnv

from train import Agent, Args


ENV_ID = "StackCube-v1"
CHECKPOINT_PATH = "/home/rakasheh/master/cassandra/maniskill/checkpoints/diffusion_state/checkpoint_diffusion_policy_stack_cube.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Must match training config
    # -----------------------------
    args = Args(
        env_id=ENV_ID,
        obs_horizon=2,
        act_horizon=8,
        pred_horizon=16,
        max_episode_steps=300,
    )

    # -----------------------------
    # Environment
    # -----------------------------
    env = gym.make(
        ENV_ID,
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
        max_episode_steps=300,
    )

    obs_dim = env.single_observation_space.shape[0]

    env.single_observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(args.obs_horizon, obs_dim),
        dtype=np.float32,
    )

    # -----------------------------
    # Agent
    # -----------------------------
    agent = Agent(env, args).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint.get("ema_agent", checkpoint)

    agent.load_state_dict(state_dict)
    agent.eval()

    # -----------------------------
    # Reset
    # -----------------------------
    obs, info = env.reset()
    obs_history = np.stack([obs] * args.obs_horizon)

    if hasattr(agent, "reset"):
        agent.reset()

    terminated = False
    truncated = False

    # -----------------------------
    # Rollout loop
    # -----------------------------
    while not (terminated or truncated):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_history).float().unsqueeze(0).to(device)
            action_seq = agent.get_action(obs_t)
            action = action_seq[0, 0].cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        obs_history = np.roll(obs_history, shift=-1, axis=0)
        obs_history[-1] = obs

        env.render()

    print("Episode finished.")
    print("Success:", info.get("success", False))

    env.close()


if __name__ == "__main__":
    main()