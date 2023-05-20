from torch_geometric.data import Dataset
from deepchem.data.pytorch_datasets import _TorchDiskDataset
from deepchem.data import DiskDataset


class AqSolDBDataset(Dataset):

    def __init__(self, deepchem_dataset: _TorchDiskDataset):
        self.graph_list = [
            mol.to_pyg_graph() for mol, _, _, _ in deepchem_dataset
        ]
        self.labels = [y for _, y, _, _ in deepchem_dataset]
        self.length = len(self.labels)
        self._indices = None

    @staticmethod
    def from_deepchem(location):
        dataset = DiskDataset(location)
        return AqSolDBDataset(dataset.make_pytorch_dataset())

    def __getitem__(self, i):
        graph = self.graph_list[i]
        label = self.labels[i]
        return graph, label

    def __len__(self):
        return self.length

    def len(self):
        return len(self)

    def get(self, idx):
        graph = self.graph_list[idx]
        label = self.labels[idx]
        return graph, label
