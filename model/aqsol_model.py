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
from torch_geometric.nn import MessagePassing
from sklearn.model_selection import KFold
from numpy import std
import wandb
from aqsol_dataset import AqSolDBDataset
import pickle
from device import get_device
import numpy as np
from torch_geometric.utils import add_self_loops


def calculate_wmse(mse, std_diff) -> float:
    return mse * (std_diff ** 2)


class SumConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(SumConv, self).__init__(aggr='add')
        hidden = (out_channels + in_channels) // 2

        self.lin1 = nn.Linear(in_channels, hidden)
        self.lin2 = nn.Linear(hidden, out_channels)

        self.in_norm = nn.BatchNorm1d(in_channels)
        self.out_norm = nn.BatchNorm1d(out_channels)

    # Messages received from adjacent nodes
    def message(self, x_j):
        return x_j

    # Aggregated

    # New tensor for this node
    def update(self, aggr_out):
        aggr_out = F.dropout(aggr_out, training=self.training)
        aggr_out = self.lin1(aggr_out).relu()
        aggr_out = F.dropout(aggr_out, training=self.training)
        aggr_out = self.lin2(aggr_out)
        return aggr_out

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.in_norm(x)
        x = self.propagate(edge_index, x=x)
        x = self.out_norm(x)
        return x


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
            "GCN": GCNConv,
            "SUM": SumConv
        }[config["architecture"]]

        hidden_channels = config["hidden_channels"]
        hidden_layers = config["hidden_layers"]

        dropout_p = config["dropout"]
        self.dropout = nn.Dropout(p=dropout_p)

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

        self.loss = nn.MSELoss()
        self.optimizer = Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        self.pooling = {
            "mean": global_mean_pool,
            "add": global_add_pool
        }[pooling]

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index

        # # Handle conv model
        mol_x = self.dropout(mol_x)
        mol_x = self.conv1(mol_x, mol_edge_index).relu()

        for conv_layer in self.conv_layers:
            mol_x = self.dropout(mol_x)
            mol_x = conv_layer(mol_x, mol_edge_index).relu()

        mol_x = self.pooling(mol_x, mol.batch)

        for lin_layer in self.lin_layers:
            mol_x = F.dropout(mol_x, training=self.training)
            mol_x = lin_layer(mol_x).relu()

        mol_x = F.dropout(mol_x, training=self.training)
        mol_x = self.lin(mol_x)

        return mol_x

    def predict(self, mol, min=-13.1719, max=2.1376816201):
        pred = self.forward(mol).detach().cpu().numpy().flatten()
        return pred * (max - min) + min


class DSM(nn.Module):

    def __init__(self):
        super(DSM, self).__init__()
        self.gcn1 = GCNConv(30, 128)
        self.gcn2 = GCNConv(128, 128 // 2)
        self.gcn3 = GCNConv(128 // 2, 128 // 4)

        self.fc1 = nn.Linear(128 // 4, 1)

        self.loss = nn.MSELoss()
        self.optimizer = Adam(
            self.parameters(),
            lr=1e-5,
            weight_decay=0.00000182366906032867)

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index
        mol_x = self.gcn1(mol_x, mol_edge_index).relu()
        mol_x = F.dropout(
            mol_x,
            p=0.2
        )
        mol_x = self.gcn2(mol_x, mol_edge_index).relu()
        mol_x = F.dropout(
            mol_x,
            p=0.2
        )
        mol_x = self.gcn3(mol_x, mol_edge_index)
        add = global_add_pool(mol_x, mol.batch)
        mol_x = torch.cat([add], dim=1)
        mol_x = self.fc1(mol_x)

        return mol_x

    def predict(self, mol, min=-13.1719, max=2.1376816201):
        self.eval()
        pred = self.forward(mol).detach().cpu().numpy().flatten()
        return pred * (max - min) + min


class Validator:

    def __init__(self, model, dataset, device):
        self.dataset = dataset
        self.model = model
        self.device = device

    def validate(self, verbose=False) -> float:
        self.model.eval()
        for batch in DataLoader(self.dataset, batch_size=len(self.dataset)):
            graphs, labels = batch
            graphs = graphs.to(self.device)

            preds = self.model(graphs).detach().cpu().numpy().flatten()
            labels = labels.detach().numpy()
            if verbose:
                print("Labels", labels, "Predictions", preds)
            evs = explained_variance_score(labels, preds)
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
        model.optimizer.zero_grad()
        epoch_loss = 0
        for batch in DataLoader(train_dataset, batch_size=self.batch_size):
            graphs, labels = batch
            graphs = graphs.to(self.device)

            labels = labels.reshape((len(labels), 1))
            labels = labels.to(self.device)

            pred = model(graphs)
            loss = model.loss(pred, labels)
            epoch_loss += loss.item()
            loss.backward()
            model.optimizer.step()
        self.mean_loss += epoch_loss
        return epoch_loss

    def run(
        self,
        validator,
        train_dataset,
        model,
        wandb_run=None,
        patience=20,
        log=True
    ):
        lowest_mse = float("inf")
        epoch_counter = 0
        not_improved_counter = 0
        epoch_loss = 0
        if wandb_run is not None:
            wandb.run = wandb_run

        while not_improved_counter < patience and epoch_counter <= 1000:
            epoch_counter += 1
            epoch_loss = self.train_one_epoch(train_dataset, model)
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
        return epoch_loss

    def run_cross_validation(self, model_class, wandb_run, model_config,
                             model_kwargs, patience):
        n_folds = 5
        skf = KFold(n_splits=n_folds, shuffle=True)
        device = get_device()
        mses = np.zeros(n_folds)
        maes = np.zeros(n_folds)
        std_diffs = np.zeros(n_folds)
        evss = np.zeros(n_folds)
        for fold, (train_index, val_index) in enumerate(
                skf.split(range(len(self.dataset)))):
            print(f"Fold {fold}")
            train_dataset = self.dataset.index_select(train_index)
            validation_dataset = self.dataset.index_select(val_index)
            print(type(validation_dataset))
            fold_model = model_class(30, model_config, **model_kwargs).to(device)
            fold_validator = Validator(fold_model, validation_dataset, device)
            loss = self.run(
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
        wandb.run = wandb_run
        wandb.log({
            "evs": evss.mean(),
            "loss": loss,
            "mse": mses.mean(),
            "mae": maes.mean(),
            "std_diff": std_diffs.mean(),
            "wmse": calculate_wmse(
                mses.mean(), std_diffs.mean()
            )
        })


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "batch_size": 32,
        # "lr": 0.000001,
        # "weight_decay": 0.00000182366906032867,
        # "architecture": "GAT",
        # "dataset": "min-max-scale",
        # "pooling": "mean",
        "patience": 69
        # "conv_hc_1": 30,
        # "conv_hc_2": 300,
        # "conv_hc_3": 210,
        # "conv_hc_4": 270,
        # "conv_do_1": 0.02624513997586908,
        # "conv_do_2": 0.3518271559744446,
        # "conv_do_3": 0.75788218327396,
        # "conv_do_4": 0.30115301272947137,
        # "lin_n_1": 30,
        # "lin_n_2": 270,
        # "lin_do_1": 0.9070104584027368,
        # "lin_do_2": 0.8803097834067043
    }
    wandb_run = wandb.init(config=config, project="SolubilityPredictor")
    model = AqSolModel(
        30,
        config,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        pooling=config["pooling"]
    ).to(device)
    # global_model: AqSolModel
    # with open("model/trained_model", "rb") as f:
    #     global_model = pickle.load(f)
    # global_model.train()
    # model = DSM()
    # model = DSM(model=global_model)
    # print(model)
    train = AqSolDBDataset.from_deepchem("data/aqsoldb_temp")
    test = AqSolDBDataset.from_deepchem("data/aqsoldb_test")
    trainer = Trainer(
        model,
        train,
        config["batch_size"],
        device
    )
    trainer.run_cross_validation(
        AqSolModel,
        wandb_run=wandb_run,
        config=config,
        patience=config["patience"]
    )
    test_validator = Validator(model, test, device)
    wandb.log(test_validator.validate())
    with open("model/small_trained_model", "wb") as f:
        pickle.dump(model, f)
