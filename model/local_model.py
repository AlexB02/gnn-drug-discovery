from torch_geometric.data import Data, Dataset
from rdkit import DataStructs
import numpy as np
import torch
from .aqsol_model import LocalModel, Trainer, Validator
from pre_processor.aqsol_preprocessor import SolubilityDataset
import wandb
from sklearn.metrics import (
    mean_squared_error, explained_variance_score, mean_absolute_error)
from utils.log import log


def generate_dataset(seed: Data,
                     top_n: int,
                     dataset: Dataset,
                     thresh: float = 0,
                     similar: bool = True) -> Dataset:
    idxs = []
    if not similar:
        idxs = [x for x in np.random.randint(0, len(dataset), size=top_n)]
    else:
        seed_fp = seed.fingerprint
        dataset_fps = [x.fingerprint for x in dataset]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(seed_fp,
                                                           dataset_fps))
        print(type(sims))
        above_threshold_idxs = np.where(sims > thresh)[0]
        if len(above_threshold_idxs) >= top_n:
            idxs = np.argsort(sims[above_threshold_idxs])[-top_n:]
        else:
            remaining = top_n - len(above_threshold_idxs)
            random_idxs = np.random.randint(0, len(dataset), size=remaining)
            idxs = np.concatenate([above_threshold_idxs, random_idxs])
    return dataset.index_select(idxs)


def generate_train_valid(dataset: Dataset, split: float) -> tuple:
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int((1 - split) * n)
    train, validation = np.split(indices, [split])
    return dataset.index_select(train), dataset.index_select(validation)


def create_local_model(smiles: str, model_loc: str):
    pass


def tune_hyperparameters(config=None):
    wandb.init(config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = wandb.config
    tests = 10
    preds = np.zeros(tests)
    labels = np.zeros(tests)
    losses = np.zeros(tests)

    train: SolubilityDataset = torch.load("data/train.pt")
    ds: SolubilityDataset = torch.load("data/valid.pt")
    log("Created datasets")

    for i, idx in enumerate(np.random.randint(0, len(ds), size=tests)):
        seed = ds[idx]
        log("Got seed")
        model = LocalModel(
            30,
            config,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            pooling=config["pooling"]
        ).to(device)
        log("Created local model")

        temp: SolubilityDataset = generate_dataset(seed,
                                                   config["dataset_size"],
                                                   train,
                                                   thresh=config["thresh"],
                                                   similar=config["tanimoto"])
        log("Generated temp dataset")
        temp_train, temp_validation = generate_train_valid(temp, 0.3)
        log("Generated train and valid datasets")

        trainer = Trainer(
            model,
            temp_train,
            config["batch_size"],
            device
        )
        log("Created trainer")
        validator = Validator(model,
                              temp_validation,
                              device)
        log("Created validator")
        losses[i], _ = trainer.run(validator,
                                   train,
                                   model,
                                   # wandb_run,
                                   patience=25,
                                   log=False)
        log("Set losses[i]")
        pred = model(seed.to(device)).detach().cpu().numpy().flatten()[0]
        log("Got pred")
        label = seed.y.cpu().numpy()
        log("Got label")
        preds[i] = pred
        log("Set pred")
        labels[i] = label
        log("Set label")
        print(i, (pred - label) ** 2)
        log("Calculated and printed")

    wandb.log(
        {
            "evs": explained_variance_score(labels, preds),
            "mse": mean_squared_error(labels, preds),
            "mae": mean_absolute_error(labels, preds),
            "loss": losses.mean()
        })
    log("Wandb logged")


def local_hyperopt():
    sweep_config = {
        "name": "Local No-Conv-DO",
        "method": "bayes",
        "metric": {
            "goal": "maximize",
            "name": "evs"
        },
        "parameters": {
            "batch_size": {
                "values": [16, 32, 64]
            },
            "lr": {
                "min": 1e-5,
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
                "values": [10, 20, 30]
            },
            "hidden_channels": {
                "min": 30,
                "max": 500
            },
            "hidden_layers": {
                "values": [0, 1]
            },
            "linear_layers": {
                "values": [0, 1]
            },
            "l_do_p": {
                "min": float(0),
                "max": float(1)
            },
            "dataset_size": {
                "min": 3,
                "max": 15
            },
            "tanimoto": {
                "values": [True]
            },
            "thresh": {
                "min": float(0),
                "max": float(1)
            }
        }
    }

    sweep_id = wandb.sweep(
        sweep_config, project="SolubilityPredictor"
    )
    wandb.agent(
        sweep_id,
        function=tune_hyperparameters,
        project="SolubilityPredictor",
        count=100
    )
    wandb.finish()
