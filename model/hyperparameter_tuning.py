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
device = torch.device("cpu")
if torch.cuda.is_available():
    cuda = int(os.getenv('CUDA_VISIBLE_DEVICES'))
    print(f"CUDA available: {cuda}")
    subprocess.call(f"cuda:{cuda}", shell=True)
    device = torch.device(f"cuda:{cuda}")
else:
    print("CUDA not available")
print(f"Loaded device: {device}")

print("Logging into wandb")
wandb.login(key="f1c8bcb101a330b26b1259276de798892fbce6a0")
print("Logged into wandb")


train_dataset = AqSolDBDataset(train.make_pytorch_dataset())
validation_dataset = AqSolDBDataset(validation.make_pytorch_dataset())
test_dataset = AqSolDBDataset(test.make_pytorch_dataset())


sweep_config = {
    "name": "sweep",
    "method": "bayes",
    "metric": {
        "goal": "minimize",
        "name": "mse"
    },
    "parameters": {
        "hidden_channels": {
            "min": 30,
            "max": 512
        },
        "num_epochs": {
            "min": 50,
            "max": 1000
        },
        "batch_size": {
            "min": 32,
            "max": 128
        },
        "lr": {
            "min": 1e-6,
            "max": 1e-3
        },
        "weight_decay": {
            "min": float(0),
            "max": 1e-5
        },
        "dropout": {
            "min": float(0),
            "max": float(1)
        },
        "n_conv_layers": {
            "min": 3,
            "max": 7
        },
        "n_lin_layers": {
            "min": 3,
            "max": 10
        }
    }
}


sweep_id = wandb.sweep(
  sweep_config, project="SolubilityPredictor"
)


def tune_hyperparameters(config=None):
    with wandb.init(config=config):
        config = wandb.config
        model = AqSolModel(
          30,
          config.hidden_channels,
          lr=config.lr,
          weight_decay=config.weight_decay,
          dropout=config.dropout,
          n_conv_layers=config.n_conv_layers
        ).to(device)
        trainer = Trainer(model, train_dataset, config.batch_size)
        trainer.run(
            config.num_epochs,
            Validator(model, validation_dataset),
            tuning=True)


wandb_config = {
    "architecture": "ConvGNN",
}


wandb.agent(
    sweep_id,
    function=tune_hyperparameters,
    project="SolubilityPredictor",
    count=30
)

# model = AqSolModel(30, 240, 0.001, 0.000001, 0.05, 5, 7)
# print(model)
# trainer = Trainer(model, train_dataset, 128)
# validator = Validator(model, validation_dataset)
# trainer.run(200, validator, earlyStopping=False)

# test_validation = Validator(model, test_dataset).validate(verbose=False)
# print(f"Test validation: {test_validation}")
