import pandas as pd
from sklearn.model_selection import train_test_split
from deepchem.trans import MinMaxTransformer
from deepchem.data import NumpyDataset
from deepchem.data import DiskDataset
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData
from numpy import array


aqsoldb = pd.read_csv("data/aqsoldb.csv")
aqsoldb = pd.DataFrame({
  "logS": aqsoldb['Solubility'],
  "SMILES": aqsoldb["SMILES"]
})

train, test_and_validation = train_test_split(aqsoldb, test_size=0.2)
validation, test = train_test_split(test_and_validation, test_size=0.5)

train = NumpyDataset(train['SMILES'], y=train['logS'])
validation = NumpyDataset(validation['SMILES'], y=validation['logS'])
test = NumpyDataset(test['SMILES'], y=test['logS'])

transformer = MinMaxTransformer(transform_y=True, dataset=train)

train = transformer.transform(train)
validation = transformer.transform(validation)
test = transformer.transform(test)


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


DiskDataset.from_numpy(
    X=train_featurized,
    y=train_y,
    data_dir="data/aqsoldb_train"
)
DiskDataset.from_numpy(
    X=validation_featurized,
    y=validation_y,
    data_dir="data/aqsoldb_valid"
)
DiskDataset.from_numpy(
    X=test_featurized,
    y=test_y,
    data_dir="data/aqsoldb_test"
)
