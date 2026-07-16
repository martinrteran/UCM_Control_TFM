To update the package use pip install -e .
poetry update
poetry install
poetry lock --no-update
wandb sync --sync-all
