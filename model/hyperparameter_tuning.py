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
    "name": "GAT",
    "method": "bayes",
    "metric": {
        "goal": "maximise",
        "name": "evs"
    },
    "parameters": {
        "conv_hc_1": {
            "min": 30,
            "max": 600
        },
        "conv_hc_2": {
            "min": 30,
            "max": 600
        },
        "conv_hc_3": {
            "min": 30,
            "max": 600
        },
        "conv_hc_4": {
            "min": 30,
            "max": 600
        },
        "conv_do_1": {
            "min": float(0),
            "max": float(1)
        },
        "conv_do_2": {
            "min": float(0),
            "max": float(1)
        },
        "conv_do_3": {
            "min": float(0),
            "max": float(1)
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "lr": {
            "min": 1e-6,
            "max": 1e-1
        },
        "weight_decay": {
            "min": float(0),
            "max": 1e-3
        },
        "pooling": {
            "values": ["mean", "add"]
        },
        "architecture": {
            "values": ["GAT", "GCN"]
        },
        "patience": {
            "values": [10, 20, 30]
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
    trainer.run(
        Validator(model, validation_dataset, device),
        tuning=True,
        wandb_run=wandb_run,
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
