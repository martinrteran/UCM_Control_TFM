import datetime
from tfm_control_ucm.core.grid_env.environment import *
from tfm_control_ucm.agents.agent import *
from tfm_control_ucm.agents.utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
exec_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
with SummaryWriter(log_dir=f"./runs/grid_agent_training/{exec_date}") as writer:
    try:
        env = Grid_Robot_Env(map=GridMap.load("./src/tfm_control_ucm/maps/map_simple.json"), 
                            robot=GridRobot(),lidar_config={}, cell_size=10)

        obs_dim = env.get_observation_dim()
        act_dim = env.get_action_dim()

        agent = GridAgent(obs_dim,act_dim,device)

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

                if loss is not None:
                    writer.add_scalar("Loss/TD_Error", loss, global_step)
                
                writer.add_scalar("Policy/Epsilon", agent.epsilon, global_step)
                writer.add_scalar("Distance/2Goal", state[-1][0], global_step)
            if episode % 100 == 0:
                agent.save(f"checkpoints/agent_ep{episode}.pth")
            writer.add_scalar("Policy_Epsilon/Episode", agent.epsilon, episode)
            writer.add_scalar("Reward/Episode", ep_reward, episode)
            writer.add_scalar("Distance/Episode", state[-1][0], episode)


        agent.save("checkpoints/agent_final.pth")
    except Exception as ex:
        print(ex)



