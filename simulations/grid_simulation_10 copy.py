from tfm_control_ucm.core.agents.agent import GridAgent
from tfm_control_ucm.core.agents.networks import GridAgentNet
from tfm_control_ucm.core.agents.utils import DQNConfig, RLAlgorithm
import tfm_control_ucm.core.training.TrainingManager as training_manager
import tfm_control_ucm.core.grid_env.environment as grid_env
import tfm_control_ucm.core.grid_env.map as maps
import tfm_control_ucm.core.grid_env.robot as robot
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
mapa_simple = maps.GridMap.load("./maps/map_obstacle_free.json")
robot_simple = robot.GridRobot()
lidar_config = {"num_rays": 30, "max_range": 10, "fov": 2*np.pi, "noise_std": 1e-3, "with_cache": True}
def reward_fn(action, obs, previous_action, previous_obs, hit_obstacle, steps, max_steps, robot_pos, goal_pos, map_diagonal, lidar_max_range):
    previous_obs = previous_obs.reshape((-1, 2))
    obs = obs.reshape((-1, 2))
    _safety_margin = 3.0
    _max_proximity_penalty = 1.0

    dist2goal = obs[-1][0]
    previous_dist2goal = previous_obs[-1][0]
    min_dist = np.min(obs[:-1, 0][obs[:-1, 0] >= 0]) if np.any(obs[:-1, 0] >= 0) else lidar_max_range

    info = {'max_steps_reached': steps >= max_steps, 'hit_obstacle': hit_obstacle, 'min_dist': min_dist, 'dist2goal': dist2goal,'steps': steps, 'goal_pos': goal_pos, 'robot_pos': robot_pos}

    # 1. Progress toward goal — the dominant signal, bounded per-step
    progress = 10.0 * (previous_dist2goal - dist2goal) / map_diagonal
    reward = progress * 1.0
    info['reward_components'] = {'progress': progress, 'step_cost': -0.05, 'proximity_penalty': 0.0, 'terminal_reward': 0.0}

    # 2. Small constant step cost so idling/short paths are preferred over long ones
    reward -= 0.05

    # 3. Bounded obstacle-proximity penalty — no division, no blow-up
    closeness = 0.0
    if min_dist < _safety_margin:
        closeness = (_safety_margin - min_dist) / _safety_margin  # in [0, 1]
        proximity_penalty = _max_proximity_penalty * closeness
        reward -= proximity_penalty
        info['reward_components']['proximity_penalty'] = -proximity_penalty


    # 5. Terminal reward/penalty — clearly bigger than any step reward, but not extreme
    terminated = False
    if dist2goal < 1:
        terminated = True
        reward += 20.0
        info['reward_components']['terminal_reward'] = 20.0
    
    if not terminated and hit_obstacle:
        reward -= 5.0
        info['reward_components']['terminal_reward'] = -5.0
        
    truncated = hit_obstacle or bool(steps >= max_steps)
    
    componente_max = abs(info['reward_components']['progress'])
    componente_min = abs(info['reward_components']['progress'])
    for component in info['reward_components']:
        if abs(info['reward_components'][component]) > componente_max:
            componente_max = abs(info['reward_components'][component])
        if abs(info['reward_components'][component]) < componente_min:
            componente_min = abs(info['reward_components'][component])
    
    info['componente_reward_min'] = componente_min
    info['componente_reward_max'] = componente_max
    info['componente_reward_diff'] = componente_max - componente_min

    return reward, terminated, truncated, info
env = grid_env.Grid_Robot_Sections_Env(map=mapa_simple, robot=robot_simple, reward_fn = reward_fn, cell_size=10, max_iteration_steps=1000, render_mode="human")

action_dim = env.get_action_dim()
obv_dim = env.get_observation_dim()
rl_config = DQNConfig(obs_dim=obv_dim, action_dim=action_dim, algorithm=RLAlgorithm.GRID_AGENT,
                     device=device,name="Grid_Agent", double_dqn = False, batch_size=64, lr=1e-3, gamma=0.99,
                     eps_start=1.0, eps_end=0.005, eps_decay=1_000_000, target_update=1000, buffer_size=100_000)

policy_net = GridAgentNet(obs_dim=obv_dim, action_dim=action_dim).to(device)
target_net = GridAgentNet(obs_dim=obv_dim, action_dim=action_dim).to(device)
agent = GridAgent(rl_config, policy_net = policy_net, target_net = target_net)
agent.load(rf"checkpoints\Test\10_v2\Grid_Agent\final.pth")

t_config = training_manager.TrainerConfig(
    eps_step=10,
    num_episodes=50_000,
)
singleTrainer = training_manager.SingleTrainer("./runs/Test/10_v2","./checkpoints/Test/10_v2",t_config)
singleTrainer.add_agents(agent)
singleTrainer.add_environments(env)

singleTrainer.simulate(agent, env, num_episodes=100, render=True)