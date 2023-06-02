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
from rdkit import Chem
from rdkit.Chem import AllChem


aqsoldb = pd.read_csv("data/aqsoldb.csv")
aqsoldb = pd.DataFrame({
  "logS": aqsoldb['Solubility'],
  "SMILES": aqsoldb["SMILES"]
})

# Break dataset into 4 groups
train_pd, extra = train_test_split(aqsoldb, test_size=0.2)

# valid_test, seeds_pd = train_test_split(extra, test_size=0.2)

validation_pd, test_pd = train_test_split(extra, test_size=0.5)


train = NumpyDataset(train_pd['SMILES'], y=train_pd['logS'])
validation = NumpyDataset(validation_pd['SMILES'], y=validation_pd['logS'])
# seeds = NumpyDataset(seeds_pd['SMILES'], y=seeds_pd['logS'])
test = NumpyDataset(test_pd['SMILES'], y=test_pd['logS'])

transformer = MinMaxTransformer(transform_y=True, dataset=train)

train = transformer.transform(train)
validation = transformer.transform(validation)
# seeds = transformer.transform(seeds)
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
        dataset.y[indices],
        indices
    )


featurizer = MolGraphConvFeaturizer(use_edges=True)

(train_featurized,
 train_y,
 train_i) = featurize_dataset(train, featurizer)
(validation_featurized,
 validation_y,
 validation_i) = featurize_dataset(validation, featurizer)
# (seeds_featurized,
#  seeds_y,
#  seeds_i) = featurize_dataset(seeds, featurizer)
(test_featurized,
 test_y,
 test_i) = featurize_dataset(test, featurizer)

train_pd.iloc[train_i].to_csv("data/train.csv")
validation_pd.iloc[validation_i].to_csv("data/validation.csv")
# seeds_pd.iloc[seeds_i].to_csv("data/seeds.csv")
test_pd.iloc[test_i].to_csv("data/test.csv")


class SolubilityDataset(Dataset):

    def __init__(self, smiles, mols, sols):
        super(SolubilityDataset, self).__init__()
        self.data = []
        for mol_smiles, mol, logs in zip(smiles, mols, sols):
            x = torch.tensor(mol.node_features).float()
            edge_index = torch.tensor(mol.edge_index)
            y = torch.tensor(logs).float()
            fingerprint = AllChem.GetMorganFingerprint(
                Chem.MolFromSmiles(mol_smiles), 2)

            self.data.append(
                Data(x=x,
                     edge_index=edge_index,
                     y=y,
                     smiles=mol_smiles,
                     fingerprint=fingerprint)
            )

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


train_torch_dataset = SolubilityDataset(train.X,
                                        train_featurized,
                                        train_y)
torch.save(train_torch_dataset, "data/train.pt")

valid_torch_dataset = SolubilityDataset(validation.X,
                                        validation_featurized,
                                        validation_y)
torch.save(valid_torch_dataset, "data/valid.pt")

# seeds_torch_dataset = SolubilityDataset(seeds.X,
#                                         seeds_featurized,
#                                         seeds_y)
# torch.save(seeds_torch_dataset, "data/seeds.pt")  # on 501

test_torch_dataset = SolubilityDataset(test.X,
                                       test_featurized,
                                       test_y)
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
