import datetime
import signal
import sys
import time
from torch import argmax
from tfm_control_ucm.core.grid_env.environment import *
from tfm_control_ucm.core.agents.agent import *
from tfm_control_ucm.core.agents.utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class InterruptHandler:
    """
    Context manager for handling keyboard interrupts with auto-save.
    - Ctrl+C → Save checkpoint and exit
    - Ctrl+\\ (SIGQUIT) → Soft reset agent (Linux/Mac only)
    """
    
    def __init__(self, agent, writer=None, save_dir="checkpoints"):
        self.agent = agent
        self.writer = writer
        self.save_dir = save_dir
        self.interrupted = False
        self.soft_reset_requested = False
        self.original_sigint = None
        self.original_sigquit = None
        
    def __enter__(self):
        self.original_sigint = signal.signal(signal.SIGINT, self._handle_exit)

        # Ctrl+\ triggers soft reset (Linux/Mac only, not available on Windows)
        if hasattr(signal, 'SIGQUIT'):
            self.original_sigquit = signal.signal(signal.SIGQUIT, self._handle_soft_reset)
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original handlers
        signal.signal(signal.SIGINT, self.original_sigint)
        if hasattr(signal, 'SIGQUIT') and self.original_sigquit:
            signal.signal(signal.SIGQUIT, self.original_sigquit)
        
        if self.interrupted:
            return True  # Suppress KeyboardInterrupt
        
        if exc_type is not None and exc_type is not KeyboardInterrupt:
            self._save_emergency(exc_type, exc_val)
            return False
            
        return False
    
    def _handle_exit(self, sig, frame):
        """Ctrl+C → save and exit."""
        self.interrupted = True
        print("\n" + "="*60)
        print("🛑 Training interrupted by user (Ctrl+C)")
        print("="*60)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        checkpoint_path = f"{self.save_dir}/agent_interrupted_{timestamp}.pth"
        
        print("Saving checkpoint...")
        try:
            self.agent.save(checkpoint_path)
            print(f"✅ Checkpoint saved to: {checkpoint_path}")
            print(f"   - Global step: {self.agent.global_step}")
            print(f"   - Epsilon: {self.agent.epsilon:.4f}")
            print(f"   - Buffer size: {len(self.agent.buffer)}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
        
        if self.writer is not None:
            self.writer.close()
        
        print("✅ Cleanup complete. Exiting...")
        print("="*60)
        sys.exit(0)

    def _handle_soft_reset(self, sig, frame):
        """Ctrl+\\ → flag a soft reset to be applied at the next episode boundary."""
        self.soft_reset_requested = True
        print("\n⚠️  Soft reset requested — will apply at end of current episode...")

    def _save_emergency(self, exc_type, exc_val):
        """Save emergency checkpoint on unexpected exception."""
        print("\n" + "="*60)
        print(f"❌ Unexpected error: {exc_type.__name__}: {exc_val}")
        print("="*60)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        checkpoint_path = f"{self.save_dir}/agent_error_{timestamp}.pth"
        
        try:
            self.agent.save(checkpoint_path)
            print(f"💾 Emergency checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Could not save emergency checkpoint: {e}")


def is_stuck(recent_actions: deque, recent_rewards: deque, episode: int) -> tuple[bool, str]:
    """
    Heuristics to detect whether the agent is stuck.

    Parameters
    ----------
    recent_actions : deque
        Last N episode-level dominant actions
    recent_rewards : deque
        Last N episode rewards
    episode : int
        Current episode number

    Returns
    -------
    (stuck: bool, reason: str)
    """
    # Don't check too early (let buffer fill first)
    if episode < 5000:
        return False, ""

    # Check 1: Agent only ever picks the same action
    if len(recent_actions) >= 50 and len(set(recent_actions)) == 1:
        return True, f"Only using action {recent_actions[0]} for 50+ episodes"

    # Check 2: Rewards are not improving and all negative
    if len(recent_rewards) >= 500:
        mean_reward = sum(recent_rewards) / len(recent_rewards)
        if mean_reward < -100.0:
            return True, f"Mean reward stuck at {mean_reward:.2f} for 500+ episodes"

    return False, ""


def do_soft_reset(agent, writer, episode, exec_date, save_dir):
    """
    Perform a soft reset: save current state, reset exploration and buffer.

    Parameters
    ----------
    agent : GridAgent_2
        The DQN agent
    writer : SummaryWriter
        TensorBoard writer
    episode : int
        Current episode (for logging)
    exec_date : str
        Run timestamp (for checkpoint naming)
    save_dir : str
        Directory to save pre-reset checkpoint
    """
    tqdm.write("\n" + "="*60)
    tqdm.write("🔄 Performing soft reset...")
    tqdm.write("="*60)

    # Save pre-reset checkpoint so you can always go back
    pre_reset_path = f"{save_dir}/{exec_date}/agent_pre_reset_ep{episode}.pth"
    agent.save(pre_reset_path)
    tqdm.write(f"💾 Pre-reset checkpoint saved: {pre_reset_path}")

    agent.soft_reset()

    # Log the reset event in TensorBoard
    writer.add_scalar("Events/SoftReset", 1.0, episode)

    tqdm.write(f"✅ Soft reset complete at episode {episode}")
    tqdm.write(f"   - Epsilon: {agent.epsilon:.4f}")
    tqdm.write(f"   - Buffer: cleared")
    tqdm.write("="*60 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    try:
        torch.rand((3,4)).to(device=device)
    except:
        device = 'cpu'
    exec_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    save_dir = "checkpoints"

    env = Grid_Robot_Sections_Env(
        map=GridMap.load("./src/tfm_control_ucm/maps/map_simple.json"), 
        robot=GridRobot(),
        cell_size=10
    )
    
    obs_dim = env.get_observation_dim()
    act_dim = env.get_action_dim()
    agent = GridAgent_2_FIXED(obs_dim, act_dim, device, double_dqn=False)
    
    print("="*60)
    print(f"🚀 Starting training")
    print(f"   Device:          {device}")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim:      {act_dim}")
    print(f"   Log dir:         ./runs/Test/02/soft_reset/GridAgent_2_FIXED")
    print(f"   Ctrl+C  → Save checkpoint and exit")
    if hasattr(signal, 'SIGQUIT'):
        print(f"   Ctrl+\\ → Manual soft reset")
    print("="*60)

    # Sliding windows for stuck detection
    recent_actions  = deque(maxlen=80)   # dominant action per episode
    recent_rewards  = deque(maxlen=600)  # episode reward
    soft_reset_count = 0

    with SummaryWriter(log_dir=f"./runs/Test/02/soft_reset/GridAgent_2_FIXED") as writer:
        with InterruptHandler(agent, writer, save_dir=save_dir) as handler:

            global_step = 0

            for episode in tqdm(range(50_000), desc="Training"):
                state, _ = env.reset()
                state_reshaped = state.reshape((obs_dim,))
                done = False
                ep_reward = 0
                ep_steps = 0
                ep_action_counts = [0] * act_dim  # count each action this episode
                select_action_time = 0.0;
                step_ev_time = 0.0;
                loss_time = 0.0;
                start_dist = state[-1][0]

                while not done:
                    # t0 = time.time()
                    action = agent.select_action(state_reshaped)
                    # select_action_time += time.time() - t0
                    # t0 = time.time()
                    next_state, reward, on_goal, truncated, info = env.step(action)
                    # step_ev_time += time.time() - t0
                    
                    next_state_reshaped = next_state.reshape((obs_dim,))

                    done = on_goal or truncated
                    agent.store(state_reshaped, action, reward, next_state_reshaped, done)
                    if ep_steps%10 == 0:
                        # t0 = time.time()
                        loss = agent.train_step()
                        # loss_time += time.time() - t0

                        if loss is not None:
                            writer.add_scalar("Loss/TD_Error", loss, global_step)

                    state_reshaped = next_state_reshaped
                    state = next_state
                    ep_reward += reward
                    ep_action_counts[action] += 1
                    global_step += 1
                    ep_steps += 1
               
                    writer.add_scalar("Policy/Epsilon/Step",   agent.epsilon,  global_step)
                    writer.add_scalar("Action/Step", action,    global_step)
                

                # ── Episode boundary ────────────────────────────────────────

                # Track dominant action and reward for stuck detection
                dominant_action = int(np.argmax(ep_action_counts))
                recent_actions.append(dominant_action)
                recent_rewards.append(ep_reward)

                # # ── Manual soft reset (Ctrl+\) ──────────────────────────────
                # if handler.soft_reset_requested:
                #     handler.soft_reset_requested = False
                #     soft_reset_count += 1
                #     tqdm.write(f"🖐  Manual soft reset requested (#{soft_reset_count})")
                #     do_soft_reset(agent, writer, episode, exec_date, save_dir)
                #     recent_actions.clear()
                #     recent_rewards.clear()

                # # ── Automatic stuck detection ───────────────────────────────
                # else:
                stuck, reason = is_stuck(recent_actions, recent_rewards, episode)
                if stuck:
                    soft_reset_count += 1
                    tqdm.write(f"\n⚠️  Agent stuck: {reason}")
                    tqdm.write(f"   Triggering automatic soft reset #{soft_reset_count}")
                    do_soft_reset(agent, writer, episode, exec_date, save_dir)
                    recent_actions.clear()
                    recent_rewards.clear()

                # ── Periodic checkpoint ─────────────────────────────────────
                if episode % 500 == 0 and episode > 0:
                    checkpoint_path = f"{save_dir}/{exec_date}/agent_ep{episode}.pth"
                    agent.save(checkpoint_path)

                # ── Episode-level TensorBoard metrics ───────────────────────
                writer.add_scalar("Policy/Epsilon/Episode",  agent.epsilon,              episode)
                writer.add_scalar("Reward/Accumulated/Episode",          ep_reward,                  episode)
                writer.add_scalar("Reward/Avg/Episode",      ep_reward / ep_steps if ep_steps > 0 else 0, episode)
                #writer.add_scalar("Action/Avg/Episode",       actions_avg / ep_steps if ep_steps > 0 else 0, episode)
                writer.add_scalar("Action/Dominant/Episode",  dominant_action,            episode)
                writer.add_scalar("Distance/Final/Episode",        state[-1][0],               episode)
                writer.add_scalar("Distance/Start/Episode",  start_dist,                 episode)
                writer.add_scalar("Distance/Diff/Episode",  start_dist - state[-1][0],  episode)
                writer.add_scalar("Steps/Episode",           ep_steps,                   episode)
                writer.add_scalar("Success/Episode",         1.0 if on_goal else 0.0,    episode)  # type: ignore
                writer.add_scalar("SoftResets/Total",        soft_reset_count,            episode)
                
                # writer.add_scalar("Time/SelectAction", select_action_time,         episode)
                # writer.add_scalar("Time/StepEnv",       step_ev_time,              episode)
                # writer.add_scalar("Time/TrainStep",     loss_time,                 episode)
                # Log action distribution
                # for i, count in enumerate(ep_action_counts):
                #     writer.add_scalar(f"Actions/Action_{i}_count", count, episode)

            print("\n" + "="*60)
            print("✅ Training completed successfully!")
            print(f"   Total soft resets: {soft_reset_count}")
            print("="*60)
            final_path = f"{save_dir}/agent_final_{exec_date}.pth"
            agent.save(final_path)
            print(f"💾 Final checkpoint saved: {final_path}")
            print("="*60)


if __name__ == "__main__":
    main()