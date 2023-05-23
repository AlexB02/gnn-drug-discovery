import pandas as pd
from deepchem.trans import MinMaxTransformer
from deepchem.data import NumpyDataset
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.data import DiskDataset
import numpy as np


train = pd.read_csv("data/train.csv")
temp = pd.read_csv("data/temp_dataset.csv")

train = NumpyDataset(train['SMILES'], y=train['logS'])
temp = NumpyDataset(temp['SMILES'], y=temp['logS'])

transformer = MinMaxTransformer(transform_y=True, dataset=train)
temp = transformer.transform(temp)

featurizer = MolGraphConvFeaturizer(use_edges=True)


def remove_pos_kwarg(mol: GraphData) -> GraphData:
    del mol.kwargs['pos']
    return mol


def featurize_dataset(dataset, featurizer) -> tuple:
    featurized = [featurizer.featurize(x)[0] for x in dataset.X]
    indices = [
        i for i, data in enumerate(featurized) if type(data) is GraphData
    ]
    return (
        np.array([remove_pos_kwarg(featurized[i]) for i in indices]),
        dataset.y[indices]
    )


temp_featurized, temp_y = featurize_dataset(temp, featurizer)


DiskDataset.from_numpy(
    X=temp_featurized,
    y=temp_y,
    data_dir="data/aqsoldb_temp"
)
