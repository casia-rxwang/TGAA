import os
import pickle
import torch
from torch_geometric.data import InMemoryDataset


class EXPWL1(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EXPWL1, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.name = 'EXPWL1'
        self.num_tasks = 2

    @property
    def raw_file_names(self):
        return ['EXPWL1.pkl']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, 'raw', 'EXPWL1.pkl'), 'rb'))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_expwl1_graph_dataset(root):
    data = EXPWL1(os.path.join(root, 'EXPWL1'))
    train, val, test = 5, 3, 2

    patch = len(data) // (train + val + test)
    train_ids = list(range(0, patch * train))
    val_ids = list(range(patch * train, patch * (train + val)))
    test_ids = list(range(patch * (train + val), len(data)))

    return data, train_ids, val_ids, test_ids
