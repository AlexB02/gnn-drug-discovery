from model.local_model import local_hyperopt, local_predict, tune_hyperparameters
# flake8: noqa
from model.aqsol_model import global_training, handle_global_predict, global_cross_validation
from pre_processor.aqsol_preprocessor import SolubilityDataset
from model.hyperparameter_tuning import global_hyperopt
import sys


def handle_local_hyperparam():
    local_hyperopt()


def handle_global_training():
    global_training()


def handle_global_hyperopt():
    global_hyperopt()
    

def handle_global_cv():
    global_cross_validation()
    

def handle_local_predict(smiles):
    local_predict(smiles)


def handle_local_testing():
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
    tune_hyperparameters(config)


if __name__ == "__main__":
    method = sys.argv[1]
    if method == "local_hyperopt":
        handle_local_hyperparam()
    elif method == "train_global_model":
        handle_global_training()
    elif method == "global_hyperopt":
        handle_global_hyperopt()
    elif method == "local_predict":
        smiles = sys.argv[2]
        pred = handle_local_predict(smiles)
        print(str(pred))
    elif method == "global_cv":
        handle_global_cv()
    elif method == "global_predict":
        smiles = sys.argv[2]
        pred = handle_global_predict(smiles)
        print(str(pred))
    elif method == "local_test":
        handle_local_testing()
    else:
        print("Error: Invalid command")
