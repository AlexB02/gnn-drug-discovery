import pandas as pd
from sklearn.model_selection import train_test_split
from deepchem.trans import MinMaxTransformer
from deepchem.data import NumpyDataset
# from deepchem.data import DiskDataset
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData
from torch_geometric.data import Data, Dataset
import torch
from numpy import array


aqsoldb = pd.read_csv("data/aqsoldb.csv")
aqsoldb = pd.DataFrame({
  "logS": aqsoldb['Solubility'],
  "SMILES": aqsoldb["SMILES"]
})

train, test_and_validation = train_test_split(aqsoldb, test_size=0.2)
validation, test = train_test_split(test_and_validation, test_size=0.5)

train.to_csv("data/train.csv")
validation.to_csv("data/validation.csv")
test.to_csv("data/test.csv")

train = NumpyDataset(train['SMILES'], y=train['logS'])
validation = NumpyDataset(validation['SMILES'], y=validation['logS'])
test = NumpyDataset(test['SMILES'], y=test['logS'])

transformer = MinMaxTransformer(transform_y=True, dataset=train)

train = transformer.transform(train)
validation = transformer.transform(validation)
test = transformer.transform(test)

with open("data/scale_data.log", "w") as f:
    f.write(f"{transformer.y_max},{transformer.y_min}")


def remove_pos_kwarg(mol: GraphData) -> GraphData:
    del mol.kwargs['pos']
    return mol


def featurize_dataset(dataset, featurizer) -> tuple:
    featurized = [featurizer.featurize(x)[0] for x in dataset.X]
    indices = [
        i for i, data in enumerate(featurized) if type(data) is GraphData
    ]
    return (
        array([remove_pos_kwarg(featurized[i]) for i in indices]),
        dataset.y[indices]
    )


featurizer = MolGraphConvFeaturizer(use_edges=True)

train_featurized, train_y = featurize_dataset(train, featurizer)
validation_featurized, validation_y = featurize_dataset(validation, featurizer)
test_featurized, test_y = featurize_dataset(test, featurizer)


class SolubilityDataset(Dataset):

    def __init__(self, mols, sols):
        super(SolubilityDataset, self).__init__()
        self.data = []
        for mol, logs in zip(mols, sols):
            x = torch.tensor(mol.node_features).float()
            edge_index = torch.tensor(mol.edge_index)
            y = torch.tensor(logs).float()

            self.data.append(
                Data(x=x, edge_index=edge_index, y=y)
            )

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


train_torch_dataset = SolubilityDataset(train_featurized, train_y)
torch.save(train_torch_dataset, "data/train.pt")
valid_torch_dataset = SolubilityDataset(validation_featurized, validation_y)
torch.save(valid_torch_dataset, "data/valid.pt")
test_torch_dataset = SolubilityDataset(test_featurized, test_y)
torch.save(test_torch_dataset, "data/test.pt")


# DiskDataset.from_numpy(
#     X=train_featurized,
#     y=train_y,
#     data_dir="data/aqsoldb_train"
# )
# DiskDataset.from_numpy(
#     X=validation_featurized,
#     y=validation_y,
#     data_dir="data/aqsoldb_valid"
# )
# DiskDataset.from_numpy(
#     X=test_featurized,
#     y=test_y,
#     data_dir="data/aqsoldb_test"
# )
