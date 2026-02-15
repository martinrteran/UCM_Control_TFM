import sys
import time

from tfm_control_ucm.core.grid_env.environment import *
from tfm_control_ucm.core.agents.agent import *
from tfm_control_ucm.core.agents.utils import *
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    try:
        torch.rand((3, 4)).to(device=device)
    except:
        device = 'cpu'

    # ── Load environment ────────────────────────────────────────────────────
    env = Grid_Robot_Env(
        map=GridMap.load("./src/tfm_control_ucm/maps/map_simple.json"),
        robot=GridRobot(),
        lidar_config={},
        cell_size=10,
        render_mode="human"          # ← enable rendering
    )

    obs_dim = env.get_observation_dim()
    act_dim = env.get_action_dim()

    # ── Load agent from checkpoint ──────────────────────────────────────────
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/agent_final_2026_02_20_23_16.pth"
    agent = GridAgent_2_FIXED(obs_dim, act_dim, device, double_dqn=False)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0  # Pure greedy — no exploration during testing

    print("=" * 60)
    print(f"🎮 Starting live evaluation")
    print(f"   Device:     {device}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Press Ctrl+C to stop")
    print("=" * 60)

    # ── Metrics ─────────────────────────────────────────────────────────────
    n_episodes   = 500
    successes    = 0
    total_steps  = 0
    total_reward = 0.0

    step_delay = 0.05   # seconds between steps — increase to slow down rendering

    try:
        for episode in tqdm(range(n_episodes), desc="Testing"):
            state, _ = env.reset()
            state_reshaped = state.reshape((obs_dim,))
            done      = False
            ep_reward = 0.0
            ep_steps  = 0

            while not done:
                env.render()
                time.sleep(step_delay)

                action = agent.select_action(state_reshaped)
                next_state, reward, on_goal, truncated, info = env.step(action)
                next_state_reshaped = next_state.reshape((obs_dim,))

                done           = on_goal or truncated
                state_reshaped = next_state_reshaped
                state          = next_state
                ep_reward     += reward
                ep_steps      += 1

            # ── Episode boundary ─────────────────────────────────────────
            successes    += int(on_goal)        # type: ignore
            total_steps  += ep_steps
            total_reward += ep_reward

            result = "✅ SUCCESS" if on_goal else "❌ TRUNCATED"   # type: ignore
            tqdm.write(f"Ep {episode:>4} | {result} | state: {state[-1][0]:>8.2f} | steps: {ep_steps:>5} | reward: {ep_reward:>8.2f}")

    except KeyboardInterrupt:
        n_episodes = episode   # type: ignore  — count only completed episodes
        print("\n⚠️  Evaluation interrupted by user")

    # ── Summary ──────────────────────────────────────────────────────────────
    if n_episodes > 0:
        print("\n" + "=" * 60)
        print("📊 Evaluation results")
        print(f"   Episodes:   {n_episodes}")
        print(f"   Successes:  {successes}  ({successes / n_episodes * 100:.1f} %)")
        print(f"   Avg reward: {total_reward / n_episodes:.3f}")
        print(f"   Avg steps:  {total_steps  / n_episodes:.1f}")
        print("=" * 60)


if __name__ == "__main__":
    main()