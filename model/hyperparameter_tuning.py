import torch
import wandb
from deepchem.data import DiskDataset
from aqsol_model import AqSolModel, Trainer, Validator
from aqsol_dataset import AqSolDBDataset
import subprocess
import os

print("Loading data")
data_dir = "data/"
train = DiskDataset(data_dir + "aqsoldb_train")  # .select(range(1))
validation = DiskDataset(data_dir + "aqsoldb_valid")
test = DiskDataset(data_dir + "aqsoldb_test")
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
    print("CUDA not available")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loaded device: {device}")

print("Logging into wandb")
wandb.login(key="f1c8bcb101a330b26b1259276de798892fbce6a0")
print("Logged into wandb")


train_dataset = AqSolDBDataset(train.make_pytorch_dataset())
validation_dataset = AqSolDBDataset(validation.make_pytorch_dataset())
test_dataset = AqSolDBDataset(test.make_pytorch_dataset())


sweep_config = {
    "name": "All n-m",
    "method": "bayes",
    "metric": {
        "goal": "maximize",
        "name": "evs"
    },
    "parameters": {
        "batch_size": {
            "values": [32, 64]
        },
        "lr": {
            "values": [1e-3, 1e-2, 1e-1]
        },
        "weight_decay": {
            "min": float(0),
            "max": 1e-1
        },
        "pooling": {
            "values": ["add", "mean"]
        },
        "architecture": {
            "values": ["SUM", "GAT", "GCN"]
        },
        "patience": {
            "values": [25]
        },
        "hidden_channels": {
            "values": [30 * i for i in range(1, 6)]
        },
        "hidden_layers": {
            "values": [0, 1, 2, 3, 4, 5, 6]
        },
        "linear_layers": {
            "values": [0, 1, 2, 3, 4, 5, 6]
        },
        "dropout": {
            "values": [0.1, 0.2, 0.3]
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
    trainer = Trainer(model, train_dataset, config.batch_size, device)
    validator = Validator(model, validation_dataset, device)
    trainer.run(
        validator,
        train_dataset,
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
    count=5
)
wandb.finish()

# model = AqSolModel(30, 240, 0.001, 0.000001, 0.05, 5, 7)
# print(model)
# trainer = Trainer(model, train_dataset, 128)
# validator = Validator(model, validation_dataset)
# trainer.run(200, validator, earlyStopping=False)

# test_validation = Validator(model, test_dataset).validate(verbose=False)
# print(f"Test validation: {test_validation}")
