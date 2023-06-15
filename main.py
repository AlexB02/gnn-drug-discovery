from model.local_model import local_hyperopt
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


if __name__ == "__main__":
    method = sys.argv[1]
    if method == "local_hyperopt":
        handle_local_hyperparam()
    elif method == "train_global_model":
        handle_global_training()
    elif method == "global_hyperopt":
        handle_global_hyperopt()
    elif method == "global_cv":
        handle_global_cv()
    elif method == "global_predict":
        smiles = sys.argv[2]
        pred = handle_global_predict(smiles)
        print(str(pred))
    else:
        print("Error: Invalid command")
