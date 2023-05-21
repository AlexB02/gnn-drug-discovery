import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import std, zeros, diff
import numpy as np
import wandb
from aqsol_dataset import AqSolDBDataset


class AqSolModel(nn.Module):
    def __init__(
            self,
            n_features,
            hidden_channels,
            lr=10**-3,
            weight_decay=10**-2.5,
            dropout=0.2,
            n_conv_layers=3,
            n_linear_layers=2
            ):
        super(AqSolModel, self).__init__()

        self.conv = GCNConv(n_features, hidden_channels)

        self.conv_layers = nn.ModuleList([
            GCNConv(
                hidden_channels,
                hidden_channels) for _ in range(n_conv_layers - 1)
        ])

        self.lin_layers = nn.ModuleList([
            nn.Linear(
                hidden_channels // i,
                hidden_channels // (i + 1)) for i in range(1, n_linear_layers)
        ])
        self.out = nn.Linear(hidden_channels // n_linear_layers, 1)

        self.loss = nn.MSELoss()
        self.optimizer = Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay)
        self.dropout = dropout

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index

        mol_x = self.conv(mol_x, mol_edge_index)
        mol_x = mol_x.relu()
        mol_x = F.dropout(mol_x, p=self.dropout, training=self.training)

        for conv_layer in self.conv_layers:
            mol_x = conv_layer(mol_x, mol_edge_index).relu()
            mol_x = F.dropout(mol_x, p=self.dropout, training=self.training)

        mol_x = global_mean_pool(mol_x, mol.batch)

        for lin_layer in self.lin_layers:
            mol_x = lin_layer(mol_x).relu()

        return self.out(mol_x)


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
            mse = mean_squared_error(labels, preds)
            mae = mean_absolute_error(labels, preds)
            std_diff = std(labels) - std(preds)
            return {
                "mse": mse,
                "std_diff": std_diff,
                "mae": mae
            }


class Trainer:

    def __init__(self, model, dataset, batch_size, device):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.mean_loss = 0
        self.run_epochs = 0

    def train_one_epoch(self):
        self.model.train()
        self.model.optimizer.zero_grad()
        epoch_loss = 0
        for batch in DataLoader(self.dataset, batch_size=self.batch_size):
            graphs, labels = batch
            graphs = graphs.to(self.device)

            labels = labels.reshape((len(labels), 1))
            labels = labels.to(self.device)

            pred = self.model(graphs)
            loss = self.model.loss(pred, labels)
            epoch_loss += loss.item()
            loss.backward()
            self.model.optimizer.step()
        self.mean_loss += epoch_loss
        return epoch_loss

    def run(
        self,
        num_epochs,
        validator,
        tuning=False,
        earlyStopping=False,
        wandb_run=None
    ):
        epoch_loss = 0
        validation_losses = zeros(num_epochs)
        stds = zeros(num_epochs)
        if wandb_run is not None:
            wandb.run = wandb_run

        for i in range(num_epochs):
            epoch_loss = self.train_one_epoch()
            validation = validator.validate()
            validation_losses[i] = validation['mse']
            stds[i] = validation['std_diff']
            if not tuning:
                print(f"loss {i}: {epoch_loss}")
                print(f"mse {i}: {validation_losses[i]}")
                print(f"std {i}: {stds[i]}")
            else:
                wandb.log({
                    "loss": epoch_loss,
                    "mse": validation['mse'],
                    "std_diff": validation['std_diff'],
                    "epoch": i + 1
                })

            # Handle early stopping
            trace_back = 20
            if earlyStopping and i > trace_back:
                validation_dec = np.all(
                    diff(
                        validation_losses[max(0, (i + 1) - trace_back):i]
                    ) <= 0)
                stds_dec = np.all(
                    diff(
                        stds[max(0, (i + 1) - trace_back):i]
                    ) <= 0)
                if validation_dec and stds_dec:
                    print("Stopping early")
                    break
        if tuning:
            validation = validator.validate()
            wandb.log({
                "loss": epoch_loss,
                "mse": validation['mse'],
                "std_diff": validation['std_diff']
            })


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = {
        "hidden_channels": 502,
        "lr": 0.00001165,
        "weight_decay": 0.000005622,
        "dropout": 0.05579,
        "n_conv_layers": 7,
        "n_lin_layers": 12,
        "num_epochs": 469,
        "batch_size": 32
    }
    wandb_run = wandb.init(config=config, project="SolubilityPredictor")
    model = AqSolModel(
        30,
        hidden_channels=config["hidden_channels"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        dropout=config["dropout"],
        n_conv_layers=config["n_conv_layers"],
        n_linear_layers=config["n_lin_layers"]
    ).to(device)
    train = AqSolDBDataset.from_deepchem("data/aqsoldb_train")
    test = AqSolDBDataset.from_deepchem("data/aqsoldb_test")
    validation = AqSolDBDataset.from_deepchem("data/aqsoldb_valid")
    trainer = Trainer(
        model,
        train,
        config["batch_size"],
        device
    )
    validator = Validator(model, validation, device)
    trainer.run(
        num_epochs=config["num_epochs"],
        validator=validator,
        tuning=True,
        wandb_run=wandb_run
    )
    test_validator = Validator(model, test, device)
    wandb.log(test_validator.validate())
