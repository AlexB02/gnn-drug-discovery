from torch_geometric.data import Data, Dataset
from rdkit import DataStructs
import numpy as np
import torch
from tqdm import tqdm
from .aqsol_model import LocalModel, Trainer, Validator, SolubilityDatasets
from pre_processor.aqsol_preprocessor import SolubilityDataset
import wandb
from sklearn.metrics import (
    mean_squared_error, explained_variance_score, mean_absolute_error)
from utils.log import log
from deepchem.feat import MolGraphConvFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr


def generate_dataset(seed: Data,
                     top_n: int,
                     dataset: Dataset,
                     thresh: float = 0,
                     similar: bool = True) -> Dataset:
    idxs = []
    if not similar:
        idxs = [x for x in np.random.randint(0, len(dataset), size=top_n)]
        return dataset.index_select(idxs), []
    else:
        seed_fp = seed.fingerprint
        dataset_fps = [x.fingerprint for x in dataset]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(seed_fp,
                                                           dataset_fps))
        idxs = np.argsort(sims)[-top_n:]
        return dataset.index_select(idxs), sims[idxs]


def generate_train_valid(dataset: Dataset, split: float) -> tuple:
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int((1 - split) * n)
    train, validation = np.split(indices, [split])
    return dataset.index_select(train), dataset.index_select(validation)


def update_tracker(tracker, progress, message):
    tracker.set_description(message)
    tracker.update(progress - tracker.n)


def create_local_model(smiles: str, model_loc: str):
    tracker = tqdm(total=100,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    update_tracker(tracker, 0, "Loading global dataset")
    config = {
        "architecture": "GAT",
        "batch_size": 64,
        "dataset_size": 12,
        "hidden_channels": 300,
        "hidden_layers": 1,
        "l_do_p": 0.02501,
        "linear_layers": 1,
        "lr": 0.02167,
        "patience": 10,
        "pooling": "mean",
        "tanimoto": True,
        "thresh": 0.8063,
        "weight_decay": 0.00000891
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train: SolubilityDataset = torch.load("data/train.pt")
    update_tracker(tracker, 20, "Featurizing SMILES")
    feat = MolGraphConvFeaturizer()
    mol = feat(smiles)[0]
    del mol.kwargs['pos']

    x = torch.tensor(mol.node_features).float()
    edge_index = torch.tensor(mol.edge_index)
    fingerprint = AllChem.GetMorganFingerprint(
        Chem.MolFromSmiles(smiles), 2)
    seed = Data(x=x,
                edge_index=edge_index,
                smiles=smiles,
                fingerprint=fingerprint)
    update_tracker(tracker, 40, "Creating local model")
    model = LocalModel(30,
                       config,
                       lr=config["lr"],
                       weight_decay=config["weight_decay"],
                       pooling=config["pooling"]).to(device)
    update_tracker(tracker, 50, "Generating local dataset")

    update_tracker(tracker, 60, "Generating train-validation split")
    temp, sims = generate_dataset(seed,
                                  config["dataset_size"],
                                  train,
                                  thresh=config["thresh"],
                                  similar=config["tanimoto"])
    temp_train, temp_validation = generate_train_valid(temp, 0.3)

    update_tracker(tracker, 70, "Creating trainer")
    trainer = Trainer(
        model,
        temp_train,
        config["batch_size"],
        device
    )
    update_tracker(tracker, 80, "Creating validator")
    validator = Validator(model,
                          temp_validation,
                          device)
    update_tracker(tracker, 90, "Training model")
    loss, _ = trainer.run(validator,
                          train,
                          model,
                          # wandb_run,
                          patience=25,
                          log=False)
    update_tracker(tracker, 95, "Saving model")
    torch.save(model, model_loc)
    update_tracker(tracker, 100, "Complete")
    return model


def local_predict(smiles):
    model = create_local_model(smiles, "local_model").cpu()
    mol = MolGraphConvFeaturizer()(smiles)[0]
    del mol.kwargs['pos']
    mol = mol.to_pyg_graph()
    pred = model.predict(mol, min=-13.1719, max=2.1376816201)[0]
    return pred


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

        temp, sims = generate_dataset(seed,
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
        log(str(preds))
        log(str(labels))

    wandb.log(
        {
            "evs": explained_variance_score(labels, preds),
            "mse": mean_squared_error(labels, preds),
            "mae": mean_absolute_error(labels, preds),
            "loss": losses.mean()
        })
    log("Wandb logged")


def local_test(config=None):
    wandb.init(config=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = wandb.config
    tests = 50
    preds = np.zeros(tests)
    labels = np.zeros(tests)
    losses = np.zeros(tests)
    squared_errors = np.zeros(tests)
    max_sims = np.zeros(tests)
    min_sims = np.zeros(tests)
    mean_sims = np.zeros(tests)

    train: SolubilityDataset = torch.load("data/train.pt")
    validation: SolubilityDataset = torch.load("data/valid.pt")
    ds: SolubilityDataset = torch.load("data/test.pt")
    log("Created datasets")
    train_valid = SolubilityDatasets((train, validation))

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

        temp, sims = generate_dataset(seed,
                                      config["dataset_size"],
                                      train_valid,
                                      thresh=config["thresh"],
                                      similar=config["tanimoto"])
        max_sims[i] = sims.max()
        min_sims[i] = sims.min()
        mean_sims[i] = sims.mean()
        log(max_sims[i])
        log(min_sims[i])
        log(mean_sims[i])
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
        pred = model(seed.to(device)).detach().cpu().numpy().flatten()[0]
        label = seed.y.cpu().numpy()
        squared_errors[i] = (pred - label)**2
        preds[i] = pred
        labels[i] = label
        log(str(i) + " " + str((pred - label) ** 2))
        log(str(preds))
        log(str(labels))
        log(str(pearsonr(max_sims, squared_errors)))
        log(str(pearsonr(min_sims, squared_errors)))
        log(str(pearsonr(mean_sims, squared_errors)))

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
