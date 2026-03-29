import gymnasium as gym
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.frame_stack import FrameStack
from train import Agent, Args
from gymnasium import spaces
import tyro

if __name__ == "__main__":
    args = tyro.cli(Args)
    ENV_ID = "StackCube-v1"
    CHECKPOINT_PATH = "/root/ManiSkill/ManiSkill/examples/baselines/diffusion_policy/runs/StackCube-v1__train__1__1774618877/checkpoints/best_eval_success_at_end.pt"
    NUM_EPISODES = 100
    OUTPUT_DIR = "eval_results"

    env_kwargs = dict(
        obs_mode="state",
        control_mode="pd_ee_delta_pos",
        render_mode="rgb_array",
    )
    env = gym.make(ENV_ID, **env_kwargs, max_episode_steps=200)

    env = RecordEpisode(
        env,
        output_dir=OUTPUT_DIR,
        save_trajectory=True,
        trajectory_name="eval_trajectories_diffusion_state",
        source_type="render"
    )

    args = Args(env_id=ENV_ID, obs_horizon=2, act_horizon=8, pred_horizon=16, max_episode_steps=300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.single_observation_space.shape[0]

    env.single_observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(args.obs_horizon, obs_dim)
    )
    agent = Agent(env, args).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    agent.load_state_dict(checkpoint["ema_agent"])
    agent.eval()

    print(f"Starting NUM_EPISODES={NUM_EPISODES} evaluation episodes...")

    successes = 0
    for i in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = False
        truncated = False
        obs_history = np.stack([obs] * args.obs_horizon)  # (obs_horizon, obs_dim)
        if hasattr(agent, "reset"):
            agent.reset()

        while not (terminated or truncated):
            with torch.no_grad():
                obs_torch = torch.from_numpy(obs_history).float().unsqueeze(0).to(device)  # (1, obs_horizon, obs_dim)
                action_seq = agent.get_action(obs_torch)
                action = action_seq[0, 0].cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)

            obs_history = np.roll(obs_history, shift=-1, axis=0)
            obs_history[-1] = obs

        print(info)
        if info.get("success", False):
            successes += 1
        print(f"Episode {i + 1}/{NUM_EPISODES} finished. Success: {info.get('success', False)}")

    print(f"Evaluation completed. Success rate: {successes}/{NUM_EPISODES} = {successes / NUM_EPISODES:.2f}")
    env.close()
