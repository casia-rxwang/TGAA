"""
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import copy
import re
from abc import ABC
from typing import Tuple

import torch
import os.path as osp

from torch_geometric.data import Dataset
from itertools import repeat
from data.sstree import SSTree, Cochain
from torch import Tensor


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class SSTreeDataset(Dataset, ABC):
    def __init__(self,
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 max_dim: int = None,
                 num_classes: int = None,
                 init_method: str = 'sum',
                 c2tree: bool = False):

        self._max_dim = max_dim
        self._num_features = [None for _ in range(max_dim + 1)]
        self._init_method = init_method
        self._c2tree = c2tree

        super(SSTreeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self._num_classes = num_classes
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None

    @property
    def max_dim(self):
        return self._max_dim

    @max_dim.setter
    def max_dim(self, value):
        self._max_dim = value

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_dir(self):
        return osp.join(self.root, f'sstree_dim{self.max_dim}_{self._init_method}')

    def num_features_in_dim(self, dim):
        if dim > self.max_dim:
            raise ValueError('`dim` {} larger than max allowed dimension {}.'.format(dim, self.max_dim))
        if self._num_features[dim] is None:
            self._look_up_num_features()
        return self._num_features[dim]

    def _look_up_num_features(self):
        for sstree in self:
            for dim in range(sstree.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = sstree.cochains[dim].num_features
                else:
                    assert self._num_features[dim] == sstree.cochains[dim].num_features

    def get_idx_split(self):
        idx_split = {'train': self.train_ids, 'valid': self.val_ids, 'test': self.test_ids}
        return idx_split


class InMemorySSTreeDataset(SSTreeDataset):
    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def __init__(self,
                 root=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 max_dim: int = None,
                 num_classes: int = None,
                 include_down_adj=False,
                 init_method=None,
                 c2tree: bool = False):
        self.include_down_adj = include_down_adj
        super(InMemorySSTreeDataset, self).__init__(root,
                                                    transform,
                                                    pre_transform,
                                                    pre_filter,
                                                    max_dim,
                                                    num_classes,
                                                    init_method=init_method,
                                                    c2tree=c2tree)
        self.data, self.slices = None, None
        self.__data_list__ = None

    def len(self):
        for dim in range(self.max_dim + 1):
            for item in self.slices[dim].values():
                return len(item) - 1
        return 0

    def get(self, idx):
        if self.__data_list__ is None:
            self.preload_to_sstree()
        data = self.__data_list__[idx]
        return copy.copy(data)

    def preload_to_sstree(self):
        if self.__data_list__ is not None:
            return
        print('preload start')
        self.__data_list__ = self.len() * [None]
        for idx in range(len(self.__data_list__)):
            retrieved = [self._get_cochain(dim, idx) for dim in range(0, self.max_dim + 1)]
            cochains = [r[0] for r in retrieved if not r[1]]

            targets = self.data['labels']
            start, end = idx, idx + 1
            if torch.is_tensor(targets):
                s = list(repeat(slice(None), targets.dim()))
                cat_dim = 0
                s[cat_dim] = slice(start, end)
            else:
                assert targets[start] is None
                s = start

            target = targets[s]

            dim = self.data['dims'][idx].item()
            assert dim == len(cochains) - 1
            data = SSTree(*cochains, y=target)

            self.__data_list__[idx] = data

        print('preload done')

    def _get_cochain(self, dim, idx) -> Tuple[Cochain, bool]:

        if dim < 0 or dim > self.max_dim:
            raise ValueError(f'The current dataset does not have cochains at dimension {dim}.')

        cochain_data = self.data[dim]
        cochain_slices = self.slices[dim]
        data = Cochain(dim)
        if cochain_data.__num_vertexs__[idx] is not None:
            data.num_vertexs = cochain_data.__num_vertexs__[idx]
        if cochain_data.__num_vertexs_up__[idx] is not None:
            data.num_vertexs_up = cochain_data.__num_vertexs_up__[idx]
        if cochain_data.__num_vertexs_down__[idx] is not None:
            data.num_vertexs_down = cochain_data.__num_vertexs_down__[idx]
        elif dim == 0:
            data.num_vertexs_down = None

        for key in cochain_data.keys:
            item, slices = cochain_data[key], cochain_slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            data[key] = None
            if start != end:
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    cat_dim = cochain_data.__cat_dim__(key, item)
                    if cat_dim is None:
                        cat_dim = 0
                    s[cat_dim] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]
        empty = (data.num_vertexs is None)

        return data, empty

    @staticmethod
    def collate(data_list, max_dim):
        def init_keys(dim, keys):
            cochain = Cochain(dim)
            for key in keys[dim]:
                cochain[key] = []
            cochain.__num_vertexs__ = []
            cochain.__num_vertexs_up__ = []
            cochain.__num_vertexs_down__ = []
            slc = {key: [0] for key in keys[dim]}
            return cochain, slc

        def collect_keys(data_list, max_dim):
            keys = {dim: set() for dim in range(0, max_dim + 1)}
            for sstree in data_list:
                for dim in keys:
                    if dim not in sstree.cochains:
                        continue
                    cochain = sstree.cochains[dim]
                    keys[dim] |= set(cochain.keys)
            return keys

        keys = collect_keys(data_list, max_dim)
        types = {}
        cat_dims = {}
        tensor_dims = {}
        data = {'labels': [], 'dims': []}
        slices = {}
        for dim in range(0, max_dim + 1):
            data[dim], slices[dim] = init_keys(dim, keys)

        for sstree in data_list:
            for dim in range(0, max_dim + 1):

                cochain = None
                if dim in sstree.cochains:
                    cochain = sstree.cochains[dim]

                for key in keys[dim]:
                    if cochain is not None and hasattr(cochain, key) and cochain[key] is not None:
                        data[dim][key].append(cochain[key])
                        if isinstance(cochain[key], Tensor) and cochain[key].dim() > 0:
                            cat_dim = cochain.__cat_dim__(key, cochain[key])
                            cat_dim = 0 if cat_dim is None else cat_dim
                            s = slices[dim][key][-1] + cochain[key].size(cat_dim)
                            if key not in cat_dims:
                                cat_dims[key] = cat_dim
                            else:
                                assert cat_dim == cat_dims[key]
                            if key not in tensor_dims:
                                tensor_dims[key] = cochain[key].dim()
                            else:
                                assert cochain[key].dim() == tensor_dims[key]
                        else:
                            s = slices[dim][key][-1] + 1
                        if key not in types:
                            types[key] = type(cochain[key])
                        else:
                            assert type(cochain[key]) is types[key]
                    else:
                        s = slices[dim][key][-1] + 0
                    slices[dim][key].append(s)

                num = None
                num_up = None
                num_down = None
                if cochain is not None:
                    if hasattr(cochain, '__num_vertexs__'):
                        num = cochain.__num_vertexs__
                    if hasattr(cochain, '__num_vertexs_up__'):
                        num_up = cochain.__num_vertexs_up__
                    if hasattr(cochain, '__num_vertexs_down__'):
                        num_down = cochain.__num_vertexs_down__
                data[dim].__num_vertexs__.append(num)
                data[dim].__num_vertexs_up__.append(num_up)
                data[dim].__num_vertexs_down__.append(num_down)

            if not hasattr(sstree, 'y'):
                sstree.y = None
            if isinstance(sstree.y, Tensor):
                assert sstree.y.size(0) == 1
            data['labels'].append(sstree.y)
            data['dims'].append(sstree.dimension)

        # Cochains
        for dim in range(0, max_dim + 1):
            for key in keys[dim]:
                if types[key] is Tensor and len(data_list) > 1:
                    if tensor_dims[key] > 0:
                        cat_dim = cat_dims[key]
                        data[dim][key] = torch.cat(data[dim][key], dim=cat_dim)
                    else:
                        data[dim][key] = torch.stack(data[dim][key])
                elif types[key] is Tensor:  # Don't duplicate attributes...
                    data[dim][key] = data[dim][key][0]
                elif types[key] is int or types[key] is float:
                    data[dim][key] = torch.tensor(data[dim][key])

                slices[dim][key] = torch.tensor(slices[dim][key], dtype=torch.long)

        # Labels and dims
        item = data['labels'][0]
        if isinstance(item, Tensor) and len(data_list) > 1:
            if item.dim() > 0:
                cat_dim = 0
                data['labels'] = torch.cat(data['labels'], dim=cat_dim)
            else:
                data['labels'] = torch.stack(data['labels'])
        elif isinstance(item, Tensor):
            data['labels'] = data['labels'][0]
        elif isinstance(item, int) or isinstance(item, float):
            data['labels'] = torch.tensor(data['labels'])
        data['dims'] = torch.tensor(data['dims'])

        return data, slices

    def copy(self, idx=None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in idx]
        dataset = copy.copy(self)
        dataset.__indices__ = None
        dataset.__data_list__ = data_list
        dataset.data, dataset.slices = self.collate(data_list)

        return dataset

    def get_split(self, split):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split {split}.')
        idx = self.get_idx_split()[split]
        if idx is None:
            raise AssertionError("No split information found.")
        if '__indices__' in dir(self) and self.__indices__ is not None:
            raise AssertionError("Cannot get the split for a subset of the original dataset.")
        return self[idx]
