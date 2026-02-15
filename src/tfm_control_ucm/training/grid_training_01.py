import wandb
from tfm_control_ucm.core.grid_env.environment import *
from tfm_control_ucm.agents.agent import *
from tfm_control_ucm.agents.utils import *
from tqdm import tqdm

# -----------------------------
# WandB SETUP
# -----------------------------
# Option A: Offline mode (no server needed)
# wandb.init(project="grid_agent", mode="offline")

# Option B: Self-hosted server
wandb.init(
    project="grid_agent",
    settings=wandb.Settings(base_url="http://localhost:8080"),
    mode = 'offline'
)

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------
# ENV + AGENT SETUP
# -----------------------------
env = Grid_Robot_Env(
    map=GridMap.load("./src/tfm_control_ucm/maps/map_simple.json"),
    robot=GridRobot(),
    lidar_config={},
    cell_size=10
)

obs_dim = env.get_observation_dim()
act_dim = env.get_action_dim()

agent = GridAgent(obs_dim, act_dim, device)

# -----------------------------
# TRAINING LOOP
# -----------------------------
global_step = 0

for episode in tqdm(range(5_000)):
    state, _ = env.reset()
    state_reshaped = state.reshape((obs_dim))
    done = False
    ep_reward = 0

    while not done:
        action = agent.select_action(state_reshaped)
        next_state, reward, on_goal, truncated, info = env.step(action)
        next_state_reshaped = next_state.reshape((obs_dim))

        done = on_goal or truncated
        agent.store(state_reshaped, action, reward, next_state_reshaped, done)
        loss = agent.train_step()

        state_reshaped = next_state_reshaped
        state = next_state
        ep_reward += reward
        global_step += 1

        # -----------------------------
        # LOGGING (LIVE)
        # -----------------------------
        log_data = {
            "Policy/Epsilon": agent.epsilon,
            "Distance/2Goal": state[-1][0]
        }

        if loss is not None:
            log_data["Loss/TD_Error"] = loss

        wandb.log(log_data, step=global_step)

    # -----------------------------
    # EPISODE-LEVEL LOGGING
    # -----------------------------
    wandb.log({
        "Reward/Episode": ep_reward,
        "Policy_Epsilon/Episode": agent.epsilon,
        "Distance/Episode": state[-1][0]
    }, step=episode)

    # -----------------------------
    # CHECKPOINTING
    # -----------------------------
    if episode % 100 == 0:
        agent.save(f"checkpoints/agent_ep{episode}.pth")

agent.save("checkpoints/agent_final.pth")