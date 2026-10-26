"""The analysis scripts read runs written by the old PPO trainers through tag aliases."""
import importlib.util
from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_old_ppo_run(log_dir):
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for episode in range(3):
            writer.add_scalar("Done/Success", episode % 2, episode)
            writer.add_scalar("Steps/Ratio", 1.5, episode)
            writer.add_scalar("Done/Reward/Episode", -1.0, episode)


def test_compare_runs_reads_old_ppo_tags(tmp_path):
    write_old_ppo_run(tmp_path)
    compare_runs = load_script("compare_runs")
    assert compare_runs.load_scalar(tmp_path, "Success/Episode") == [0.0, 1.0, 0.0]
    assert compare_runs.load_scalar(tmp_path, "Steps/Ratio/Episode") == [1.5, 1.5, 1.5]
    assert compare_runs.load_scalar(tmp_path, "Loss/Episode") is None


def test_plot_runs_reads_old_ppo_tags(tmp_path):
    pytest.importorskip("matplotlib")
    write_old_ppo_run(tmp_path)
    plot_runs = load_script("plot_runs")
    data = plot_runs.load_run(tmp_path)
    assert data["Success/Episode"][1] == [0.0, 1.0, 0.0]
    assert data["Reward/Total"][1] == [-1.0, -1.0, -1.0]


def test_eval_agent_loads_recurrent_checkpoints(tmp_path):
    import torch
    from tfm_control_ucm.core.agents.agent import RecurrentPPOAgent
    from tfm_control_ucm.core.agents.networks import RecurrentGRUPolicy, RecurrentGRUValue
    from tfm_control_ucm.core.agents.utils import PPORNNConfig, RLAlgorithm

    config = PPORNNConfig(name="gru", obs_dim=18, action_dim=4, algorithm=RLAlgorithm.PPO, device="cpu", seq_len=8)
    agent = RecurrentPPOAgent(config, RecurrentGRUPolicy(18, 12, 4), RecurrentGRUValue(18, 12))
    agent.save(str(tmp_path / "final.pth"))

    eval_agent = load_script("eval_agent")
    loaded, obs_dim, action_dim = eval_agent.load_agent_from_checkpoint(str(tmp_path / "final.pth"), torch.device("cpu"))
    assert isinstance(loaded, RecurrentPPOAgent) and (obs_dim, action_dim) == (18, 4)
    assert loaded.policy_net.hidden_dim == 12
    loaded.on_episode_start()
    assert 0 <= loaded.act(torch.zeros(18).numpy(), training=False) < 4
