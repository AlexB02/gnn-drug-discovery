import torch
from torch_geometric.data import Data, Dataset
import wandb
# from deepchem.data import DiskDataset
from aqsol_model import AqSolModel, Trainer, Validator
# from aqsol_dataset import AqSolDBDataset
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


print("Loading data")
data_dir = "data/"
# train = DiskDataset(data_dir + "aqsoldb_train_s")  # .select(range(1))
# validation = DiskDataset(data_dir + "aqsoldb_valid_s")
# test = DiskDataset(data_dir + "aqsoldb_test_s")
train = torch.load(data_dir + "train.pt")
validation = torch.load(data_dir + "valid.pt")
test = torch.load(data_dir + "test.pt")
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


# train_dataset = AqSolDBDataset(train.make_pytorch_dataset())
# validation_dataset = AqSolDBDataset(validation.make_pytorch_dataset())
# test_dataset = AqSolDBDataset(test.make_pytorch_dataset())


sweep_config = {
    "name": "All n-m",
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
            "values": ["mean", "add"]
        },
        "architecture": {
            "values": ["GCN", "GAT"]
        },
        "patience": {
            "values": [10, 20, 30, 40, 50]
        },
        "hidden_channels": {
            "min": 30,
            "max": 500
        },
        "hidden_layers": {
            "min": 0,
            "max": 8
        },
        "linear_layers": {
            "min": 0,
            "max": 6
        },
        "c_do_p": {
            "min": float(0),
            "max": float(1)
        },
        "l_do_p": {
            "min": float(0),
            "max": float(1)
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


wandb_config = {
    "architecture": "ConvGNN",
}


wandb.agent(
    sweep_id,
    function=tune_hyperparameters,
    project="SolubilityPredictor",
    count=30
)
wandb.finish()

# model = AqSolModel(30, 240, 0.001, 0.000001, 0.05, 5, 7)
# print(model)
# trainer = Trainer(model, train_dataset, 128)
# validator = Validator(model, validation_dataset)
# trainer.run(200, validator, earlyStopping=False)

# test_validation = Validator(model, test_dataset).validate(verbose=False)
# print(f"Test validation: {test_validation}")
