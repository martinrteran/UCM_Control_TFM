import torch
from torch import nn
from torch.optim import Adam

from torchrl.envs import GymEnv
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.modules import MLP
from torchrl.objectives import DQNLoss
from torchrl.collectors import SyncDataCollector

from tfm_control_ucm.core.grid_env.environment import RobotEnv
from tfm_control_ucm.core.gridmap import GridMap


def main():

    # -----------------------------
    # 1. Create environment
    # -----------------------------
    gm = GridMap(size=(10, 10), obstacles=[(3,3), (4,3), (5,3)])
    env = GymEnv(
        RobotEnv(gridmap=gm, start_pos=(5, 5)),
        from_pixels=False,
        allow_done=True
    )

    obs_dim = env.observation_spec.shape[0]
    action_dim = env.action_spec.space.n

    # -----------------------------
    # 2. Q-network
    # -----------------------------
    qnet = MLP(
        in_features=obs_dim,
        out_features=action_dim,
        depth=2,
        num_cells=128,
        activation_class=nn.ReLU
    )

    optimizer = Adam(qnet.parameters(), lr=1e-3)
    loss_fn = DQNLoss(qnet)

    # -----------------------------
    # 3. Replay buffer
    # -----------------------------
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(50_000),
        batch_size=64
    )

    # -----------------------------
    # 4. Collector
    # -----------------------------
    collector = SyncDataCollector(
        env,
        policy=qnet,
        frames_per_batch=100,
        total_frames=20_000,
        exploration_type="epsilon_greedy",
        epsilon=0.1
    )

    # -----------------------------
    # 5. Training loop
    # -----------------------------
    step = 0
    for batch in collector:
        step += batch.numel()
        buffer.extend(batch)

        if len(buffer) > 1000:
            sample = buffer.sample()
            loss = loss_fn(sample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Step {step} | Batch {batch.numel()} | Loss {float(loss):.4f}")

        # -----------------------------
        # Save intermediate checkpoints
        # -----------------------------
        if step % 5000 == 0:
            torch.save(qnet.state_dict(), f"models/dqn_checkpoint_{step}.pt")

    # -----------------------------
    # Save final model
    # -----------------------------
    torch.save(qnet.state_dict(), "models/dqn_final.pt")
    print("Training finished! Model saved to models/dqn_final.pt")


if __name__ == "__main__":
    main()