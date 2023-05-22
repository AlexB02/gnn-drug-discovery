import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy import std
import wandb
from aqsol_dataset import AqSolDBDataset
import pickle


def calculate_wmse(mse, std_diff) -> float:
    return mse * (std_diff ** 2)


class AqSolModel(nn.Module):
    def __init__(
            self,
            n_features,
            hidden_channels,
            lr=10**-3,
            weight_decay=10**-2.5,
            dropout=0.2,
            n_conv_layers=3,
            n_linear_layers=2,
            pooling="mean",
            architecture="GAT"
            ):
        super(AqSolModel, self).__init__()

        self.arch = {
            "GAT": GATv2Conv,
            "GCN": GCNConv
        }[architecture]

        self.conv = self.arch(n_features, hidden_channels)

        self.conv_layers = nn.ModuleList([
            self.arch(
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
        self.pooling = {
            "mean": global_mean_pool,
            "add": global_add_pool
        }[pooling]

    def forward(self, mol):
        mol_x, mol_edge_index = mol.x, mol.edge_index

        mol_x = self.conv(mol_x, mol_edge_index)
        mol_x = mol_x.relu()
        mol_x = F.dropout(mol_x, p=self.dropout, training=self.training)

        for conv_layer in self.conv_layers:
            mol_x = conv_layer(mol_x, mol_edge_index).relu()
            mol_x = F.dropout(mol_x, p=self.dropout, training=self.training)

        mol_x = self.pooling(mol_x, mol.batch)

        for lin_layer in self.lin_layers:
            mol_x = lin_layer(mol_x).relu()

        return self.out(mol_x)

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
        lowest_mse = float("inf")
        epoch_counter = 0
        not_improved_counter = 0
        if wandb_run is not None:
            wandb.run = wandb_run

        while not_improved_counter < 20:
            epoch_counter += 1
            epoch_loss = self.train_one_epoch()
            validation = validator.validate()
            wandb.log({
                "loss": epoch_loss,
                "mse": validation['mse'],
                "std_diff": validation['std_diff'],
                "epoch": epoch_counter + 1,
                "wmse": calculate_wmse(
                    validation['mse'], validation['std_diff']
                )
            })
            if validation["mse"] < lowest_mse:
                lowest_mse = validation["mse"]
                not_improved_counter = 0
            else:
                not_improved_counter += 1

        if tuning:
            validation = validator.validate()
            wandb.log({
                "loss": epoch_loss,
                "mse": validation['mse'],
                "std_diff": validation['std_diff'],
                "wmse": calculate_wmse(
                    validation['mse'], validation['std_diff']
                )
            })


if __name__ == "__main__":
    config = {
        "batch_size": 16,
        "dropout": 0.0391,
        "hidden_channels": 205,
        "lr": 0.00003143,
        "weight_decay": 5.071e-7,
        "n_conv_layers": 1,
        "n_lin_layers": 3,
        "num_epochs": 100,
        "architecture": "GATv2 SAG",
        "dataset": "min-max-scale"
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
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    with open("model/trained_model", "wb") as f:
        pickle.dump(model, f)
