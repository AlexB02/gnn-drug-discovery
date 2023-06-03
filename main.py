from model.local_model import local_hyperopt
# flake8: noqa
from model.aqsol_model import global_training, SolubilityDataset
from model.hyperparameter_tuning import global_hyperopt
import sys


def handle_local_hyperparam():
    local_hyperopt()


def handle_global_training():
    global_training()


def handle_global_hyperopt():
    global_hyperopt()


if __name__ == "__main__":
    method = sys.argv[1]
    if method == "local_hyperopt":
        handle_local_hyperparam()
    elif method == "train_global_model":
        handle_global_training()
    elif method == "global_hyperopt":
        handle_global_hyperopt()
