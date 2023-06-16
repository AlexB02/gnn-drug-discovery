import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import (
    global_add_pool, global_mean_pool, GCNConv, GATConv)
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, explained_variance_score
)
from sklearn.model_selection import KFold
from numpy import std
import wandb
# from aqsol_dataset import AqSolDBDataset
import pickle
from .device import get_device
import numpy as np
from torch_geometric.data import Dataset
from deepchem.feat import MolGraphConvFeaturizer
from utils.log import log


def calculate_wmse(mse, std_diff) -> float:
    return mse * (std_diff ** 2)


class AqSolModel(nn.Module):
    def __init__(
            self,
            n_features,
            config,
            lr=10**-3,
            weight_decay=10**-2.5,
            pooling="mean",
            ):
        super(AqSolModel, self).__init__()

        self.config = config

        self.arch = {
            "GAT": GATConv,
            "GCN": GCNConv
        }[config["architecture"]]

        self.conv1 = self.arch(n_features, config["conv_hc_1"])
        self.conv_do_1 = nn.Dropout(p=config["conv_do_1"])

        self.conv2 = self.arch(config["conv_hc_1"], config["conv_hc_2"])
        self.conv_do_2 = nn.Dropout(p=config["conv_do_2"])

        self.conv3 = self.arch(config["conv_hc_2"], config["conv_hc_3"])
        self.conv_do_3 = nn.Dropout(p=config["conv_do_3"])

        self.conv4 = self.arch(config["conv_hc_3"], config["conv_hc_4"])

        self.lin1 = nn.Linear(config["conv_hc_4"], config["lin_n_1"])
        self.lin_do_1 = nn.Dropout(p=config["lin_do_1"])

        self.lin2 = nn.Linear(config["lin_n_1"], config["lin_n_2"])
        self.lin_do_2 = nn.Dropout(p=config["lin_do_2"])

        self.lin3 = nn.Linear(config["lin_n_2"], 1)

        self.pooling = {
            "mean": global_mean_pool,
            "add": global_add_pool
        }[pooling]

        self.optimizer = Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay)

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index

        mol_x = self.conv1(mol_x, mol_edge_index).relu()
        mol_x = self.conv_do_1(mol_x)

        mol_x = self.conv2(mol_x, mol_edge_index).relu()
        mol_x = self.conv_do_2(mol_x)

        mol_x = self.conv3(mol_x, mol_edge_index).relu()
        mol_x = self.conv_do_3(mol_x)

        mol_x = self.conv4(mol_x, mol_edge_index).relu()

        mol_x = self.pooling(mol_x, mol.batch)

        mol_x = self.lin1(mol_x).relu()

        mol_x = self.lin_do_1(mol_x)
        mol_x = self.lin2(mol_x).relu()

        mol_x = self.lin_do_2(mol_x)
        mol_x = self.lin3(mol_x)

        return mol_x

    def predict(self, mol, min=-13.1719, max=2.1376816201):
        pred = self.forward(mol).detach().cpu().numpy().flatten()
        return pred * (max - min) + min


class LocalModel(nn.Module):
    def __init__(
            self,
            n_features,
            config,
            lr=10**-3,
            weight_decay=10**-2.5,
            pooling="mean",
            ):
        super(LocalModel, self).__init__()

        self.config = config

        self.arch = {
            "GAT": GATConv,
            "GCN": GCNConv
        }[config["architecture"]]

        hidden_channels = config["hidden_channels"]
        hidden_layers = config["hidden_layers"]

        self.l_do_p = config["l_do_p"]

        self.conv1 = self.arch(n_features, hidden_channels)

        self.conv_layers = nn.ModuleList([
            self.arch(hidden_channels,
                      hidden_channels) for _ in range(hidden_layers)
        ])

        linear_layers = config["linear_layers"]
        self.lin_layers = nn.ModuleList([
            nn.Linear(hidden_channels,
                      hidden_channels) for _ in range(linear_layers)
        ])
        self.lin = nn.Linear(hidden_channels, 1)

        self.pooling = {
            "mean": global_mean_pool,
            "add": global_add_pool
        }[pooling]

        self.optimizer = Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay)

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index

        mol_x = self.conv1(mol_x, mol_edge_index).sigmoid()

        for conv_layer in self.conv_layers:
            mol_x = conv_layer(mol_x, mol_edge_index).sigmoid()

        mol_x = self.pooling(mol_x, mol.batch)

        for lin_layer in self.lin_layers:
            mol_x = F.dropout(mol_x, training=self.training, p=self.l_do_p)
            mol_x = lin_layer(mol_x).relu()

        mol_x = F.dropout(mol_x, training=self.training, p=self.l_do_p)
        mol_x = self.lin(mol_x)

        return mol_x

    def predict(self, mol, min=-13.1719, max=2.1376816201):
        pred = self.forward(mol).detach().cpu().numpy().flatten()
        return pred * (max - min) + min


class Validator:

    def __init__(self, model, dataset, device):
        self.dataset = dataset
        self.model = model
        self.device = device

    def validate(self, verbose=False) -> float:
        self.model.eval()
        for data in DataLoader(self.dataset, batch_size=len(self.dataset)):
            graphs, labels = data.to(self.device), data.y

            preds = self.model(graphs)
            preds = preds.detach().cpu().numpy().flatten()
            labels = labels.detach().cpu().numpy()

            if verbose:
                print("Labels", labels, "Predictions", preds)
            evs = explained_variance_score(labels,
                                           preds,
                                           multioutput="raw_values")
            mse = mean_squared_error(labels, preds)
            mae = mean_absolute_error(labels, preds)
            std_diff = std(labels) - std(preds)
            return {
                "evs": evs,
                "mse": mse,
                "std_diff": std_diff,
                "mae": mae,
                "wmse": calculate_wmse(mse, std_diff)
            }


class Trainer:

    def __init__(self, model, dataset, batch_size, device):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.mean_loss = 0
        self.run_epochs = 0

    def train_one_epoch(self, train_dataset, model):
        model.train()
        epoch_losses = []
        epoch_true = []
        epoch_pred = []
        criterion = torch.nn.MSELoss()

        for data in DataLoader(train_dataset,
                               batch_size=self.batch_size):
            data = data.to(self.device)
            graphs, labels = data, data.y

            model.optimizer.zero_grad()
            preds = model(graphs)
            loss = criterion(preds, labels.reshape(-1, 1).type_as(preds))

            y_true = labels.cpu().detach().numpy()
            y_pred = preds.cpu().detach().numpy().flatten()

            epoch_true.extend(list(y_true))
            epoch_pred.extend(list(y_pred))

            epoch_losses.append(loss.item())
            loss.backward()
            model.optimizer.step()

        epoch_evs = explained_variance_score(
            epoch_true, epoch_pred, multioutput="raw_values"
        )
        return sum(epoch_losses) / len(epoch_losses), epoch_evs

    def train_n_epochs(self, train_dataset, model, n_epochs) -> np.ndarray:
        epoch_losses = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            epoch_loss, _evs = self.train_one_epoch(train_dataset, model)
            epoch_losses[epoch] = epoch_loss
        return epoch_losses

    def run(
        self,
        validator,
        train_dataset,
        model,
        wandb_run=None,
        patience=20,
        log=True
    ) -> float:
        """Returns last epoch's loss"""
        # print("Training model")
        # print(model)
        lowest_mse = float("inf")
        epoch_counter = 0
        not_improved_counter = 0
        epoch_loss = 0
        if wandb_run is not None:
            wandb.run = wandb_run

        while not_improved_counter < patience and epoch_counter <= 1000:
            epoch_counter += 1
            epoch_loss, evs = self.train_one_epoch(train_dataset, model)
            # print("Epoch EVS", evs)
            validation = validator.validate()
            if log:
                wandb.log({"epoch": epoch_counter + 1}, commit=False)
                wandb.log({
                    "evs": validation['evs'],
                    "loss": epoch_loss,
                    "mse": validation['mse'],
                    "mae": validation['mae'],
                    "std_diff": validation['std_diff'],
                    "wmse": calculate_wmse(
                        validation['mse'], validation['std_diff']
                    )
                }, commit=True)
            if validation["mse"] < lowest_mse:
                lowest_mse = validation["mse"]
                not_improved_counter = 0
            else:
                not_improved_counter += 1
        return epoch_loss, epoch_counter

    def run_cross_validation(self, model_class, wandb_run, model_config,
                             patience, model_kwargs={}):
        n_folds = 5
        skf = KFold(n_splits=n_folds, shuffle=True)
        device = get_device()
        mses = np.zeros(n_folds)
        maes = np.zeros(n_folds)
        std_diffs = np.zeros(n_folds)
        evss = np.zeros(n_folds)
        epochs = np.zeros(n_folds)
        for fold, (train_index, val_index) in enumerate(
                skf.split(range(len(self.dataset)))):
            print(f"Fold {fold}")
            train_dataset = self.dataset.index_select(train_index)
            validation_dataset = self.dataset.index_select(val_index)
            print(len(train_dataset), len(validation_dataset))
            fold_model = model_class(30,
                                     model_config,
                                     **model_kwargs).to(device)
            fold_validator = Validator(fold_model, validation_dataset, device)
            loss, num_epochs = self.run(
                fold_validator,
                train_dataset,
                fold_model,
                wandb_run,
                patience,
                log=False
            )
            validation = fold_validator.validate()
            mses[fold] = validation["mse"]
            maes[fold] = validation["mae"]
            std_diffs[fold] = validation["std_diff"]
            evss[fold] = validation["evs"]
            epochs[fold] = num_epochs
            log(str(evss))
            log(str(mses))
            log(str(maes))
            log(str(std_diffs))
            log(str(epochs))
        wandb.run = wandb_run
        wandb.log({
            "evs": evss.mean(),
            "loss": loss,
            "mse": mses.mean(),
            "mae": maes.mean(),
            "std_diff": std_diffs.mean(),
            "wmse": calculate_wmse(
                mses.mean(), std_diffs.mean()
            ),
            "epoch": epochs.mean()
        })


class SolubilityDatasets(Dataset):

    def __init__(self, datasets):
        super(SolubilityDatasets, self).__init__()
        self.data = []
        for dataset in datasets:
            print(dataset)
            for elem in dataset:
                self.data.append(elem)

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


def handle_global_predict(smiles):
    global_model = None
    with open("model/global_model", "rb") as f:
        global_model = pickle.load(f)
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    mol = featurizer.featurize(smiles)[0]
    del mol.kwargs['pos']
    pred = global_model.predict(mol.to_pyg_graph(),
                                min=-13.1719,
                                max=2.1376816201)[0]
    return pred


def global_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 64,
        "lr": 0.00013850654318564852,
        "weight_decay": 0.00000720831334769755,
        "pooling": "add",
        "architecture": "GCN",
        "patience": 30,
        "conv_hc_1": 69,
        "conv_hc_2": 208,
        "conv_hc_3": 158,
        "conv_hc_4": 122,
        "conv_do_1": 0.012940682270939608,
        "conv_do_2": 0.1536591043086343,
        "conv_do_3": 0.1619722720722499,
        "lin_do_1": 0.20680253869050996,
        "lin_do_2": 0.0858066355023579,
        "lin_n_1": 112,
        "lin_n_2": 71,
    }
    wandb.init(config=config, project="SolubilityPredictor")
    model = AqSolModel(
        30,
        config,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        pooling=config["pooling"]
    ).to(device)

    train = torch.load("data/train.pt")
    validation = torch.load("data/valid.pt")
    test = torch.load("data/test.pt")
    train_valid = SolubilityDatasets((train, validation))
    print("Dataset: " + str(train_valid.len()))
    trainer = Trainer(
        model,
        train_valid,
        config["batch_size"],
        device
    )
    trainer.train_n_epochs(train_valid, model, 181)

    test_validator = Validator(model, test, device)
    wandb.log(test_validator.validate())
    with open("model/global_model", "wb") as f:
        pickle.dump(model, f)


def global_cross_validation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 64,
        "lr": 0.00014548456975291318,
        "weight_decay": 0.00000480762841983841,
        "pooling": "add",
        "architecture": "GCN",
        "patience": 30,
        "conv_hc_1": 123,
        "conv_hc_2": 224,
        "conv_hc_3": 241,
        "conv_hc_4": 150,
        "conv_do_1": 0.022601527797254908,
        "conv_do_2": 0.051709104936469234,
        "conv_do_3": 0.050563285495162535,
        "lin_do_1": 0.46493512584579194,
        "lin_do_2": 0.17310412543366466,
        "lin_n_1": 64,
        "lin_n_2": 128,
    }
    wandb_run = wandb.init(config=config, project="SolubilityPredictor")
    model = AqSolModel(
        30,
        config,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        pooling=config["pooling"]
    ).to(device)

    train = torch.load("data/train.pt")
    valid = torch.load("data/valid.pt")
    # test = torch.load("data/test.pt")
    train_valid = SolubilityDatasets((train, valid))
    print("Dataset: " + str(train_valid.len()))
    trainer = Trainer(
        model,
        train_valid,
        config["batch_size"],
        device
    )
    model_kwargs = {
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "pooling": config["pooling"]
    }
    trainer.run_cross_validation(AqSolModel,
                                 wandb_run,
                                 config,
                                 patience=config["patience"],
                                 model_kwargs=model_kwargs)
