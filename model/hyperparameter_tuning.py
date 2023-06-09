import torch
from torch_geometric.data import Data, Dataset
import wandb
from .aqsol_model import AqSolModel, Trainer, Validator
import subprocess
import os


class SolubilityDataset(Dataset):

    def __init__(self, mols, sols):
        super(SolubilityDataset, self).__init__()
        self.data = []
        for mol, logs in zip(mols, sols):
            x = torch.tensor(mol.node_features).float()
            edge_index = torch.tensor(mol.edge_index)
            y = torch.tensor(logs).float()

            self.data.append(
                Data(x=x, edge_index=edge_index, y=y)
            )

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


def global_hyperopt():
    print("Loading data")
    data_dir = "data/"
    train = torch.load(data_dir + "train.pt")
    validation = torch.load(data_dir + "valid.pt")
    print("Loaded data")
    print("Loading device")
    subprocess.call("echo Checking CUDA available", shell=True)
    if torch.cuda.is_available():
        cuda = int(os.getenv('CUDA_VISIBLE_DEVICES'))
        print(f"CUDA available: {cuda}")
        subprocess.call(f"echo {torch.cuda.device_count()}", shell=True)
        subprocess.call(f"echo {torch.cuda.get_device_name()}", shell=True)
    else:
        subprocess.call("echo CUDA not available", shell=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    wandb.login(key="f1c8bcb101a330b26b1259276de798892fbce6a0")

    sweep_config = {
        "name": "Global 4-3",
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "evs"
        },
        "parameters": {
            "batch_size": {
                "values": [64]
            },
            "lr": {
                "min": 1e-3,
                "max": 1e-1
            },
            "weight_decay": {
                "min": 0.0,
                "max": 1e-5
            },
            "pooling": {
                "values": ["add"]
            },
            "architecture": {
                "values": ["GCN"]
            },
            "patience": {
                "values": [30]
            },
            "conv_hc_1": {
                "min": 50,
                "max": 150
            },
            "conv_hc_2": {
                "min": 100,
                "max": 250
            },
            "conv_hc_3": {
                "min": 100,
                "max": 250
            },
            "conv_hc_4": {
                "min": 50,
                "max": 150
            },
            "conv_do_1": {
                "min": float(0),
                "max": float(0.2)
            },
            "conv_do_2": {
                "min": float(0),
                "max": float(0.2)
            },
            "conv_do_3": {
                "min": float(0),
                "max": float(0.2)
            },
            "conv_do_4": {
                "min": float(0),
                "max": float(0.2)
            },
            "lin_do_1": {
                "min": float(0),
                "max": float(0.5)
            },
            "lin_do_2": {
                "min": float(0),
                "max": float(0.5)
            },
            "lin_do_3": {
                "min": float(0),
                "max": float(0.5)
            },
            "lin_n_1": {
                "min": 50,
                "max": 150
            },
            "lin_n_2": {
                "min": 50,
                "max": 150
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config, project="SolubilityPredictor"
    )

    def tune_hyperparameters(config=None):
        wandb_run = wandb.init(config=config)
        config = wandb.config
        model = AqSolModel(
            30,
            config,
            lr=config.lr,
            weight_decay=config.weight_decay,
            pooling=config.pooling
        ).to(device)
        trainer = Trainer(model, train, config.batch_size, device)
        validator = Validator(model, validation, device)
        trainer.run(
            validator,
            train,
            model,
            wandb_run,
            patience=config.patience
        )

    wandb.agent(
        sweep_id,
        function=tune_hyperparameters,
        project="SolubilityPredictor",
        count=420
    )
    wandb.finish()
