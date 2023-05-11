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

import torch
import logging
import copy

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj
from typing import List


class MPParams:
    def __init__(self, batch_size: int, x: Tensor, x_num: List, x_batch: Tensor, x_index: Tensor, boundary_index: Tensor,
                 up_index: Tensor, shared_coboundaries: Tensor, down_index: Tensor, shared_boundaries: Tensor, **kwargs):
        self.batch_size = batch_size
        self.x_num = x_num
        self.max_num = max(x_num)  # x_num must not be None
        self.x_batch = x_batch

        self.x = x
        self.x_idx = x_index[0], x_index[1]  # x_index must not be None
        self.x_mask = None

        self.boundary_index = boundary_index
        self.boundary_adj = None
        self.boundary_attr = None
        self.boundary_attr_mask = None
        self.boundary_attr_idx = None

        self.up_index = up_index
        self.shared_coboundaries = shared_coboundaries
        self.up_adj = None
        self.up_x_i_idx = None
        self.up_x_j_idx = None
        self.up_attr_idx = None
        self.up_attr = None

        self.down_index = down_index
        self.shared_boundaries = shared_boundaries
        self.down_adj = None
        self.down_x_i_idx = None
        self.down_x_j_idx = None
        self.down_attr_idx = None
        self.down_attr = None

        self.kwargs = kwargs


class Cochain(object):
    def __init__(self,
                 dim: int,
                 x: Tensor = None,
                 upper_index: Adj = None,
                 lower_index: Adj = None,
                 shared_boundaries: Tensor = None,
                 shared_coboundaries: Tensor = None,
                 mapping: Tensor = None,
                 boundary_index: Adj = None,
                 upper_orient=None,
                 lower_orient=None,
                 y=None,
                 **kwargs):
        if dim == 0:
            assert lower_index is None
            assert shared_boundaries is None
            assert boundary_index is None

        self.__dim__ = dim
        self.__x = x
        self.upper_index = upper_index
        self.lower_index = lower_index
        self.boundary_index = boundary_index
        self.y = y
        self.shared_boundaries = shared_boundaries
        self.shared_coboundaries = shared_coboundaries
        self.upper_orient = upper_orient
        self.lower_orient = lower_orient
        self.__oriented__ = False
        self.__hodge_laplacian__ = None
        self.__mapping = mapping
        for key, item in kwargs.items():
            if key == 'num_vertexs':
                self.__num_vertexs__ = item
            elif key == 'num_vertexs_down':
                self.num_vertexs_down = item
            elif key == 'num_vertexs_up':
                self.num_vertexs_up = item
            else:
                self[key] = item

    @property
    def dim(self):
        return self.__dim__

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, new_x):
        if new_x is None:
            logging.warning("Cochain features were set to None. ")
        else:
            assert self.num_vertexs == len(new_x)
        self.__x = new_x

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.keys

    def __cat_dim__(self, key, value):
        if key in ['upper_index', 'lower_index', 'shared_boundaries', 'shared_coboundaries', 'boundary_index']:
            return -1
        elif isinstance(value, SparseTensor):
            return (0, 1)
        return 0

    def __inc__(self, key, value):
        if key in ['upper_index', 'lower_index']:
            inc = self.num_vertexs
        elif key in ['shared_boundaries']:
            inc = self.num_vertexs_down
        elif key == 'shared_coboundaries':
            inc = self.num_vertexs_up
        elif key == 'boundary_index':
            boundary_inc = self.num_vertexs_down if self.num_vertexs_down is not None else 0
            vertex_inc = self.num_vertexs if self.num_vertexs is not None else 0
            inc = [[boundary_inc], [vertex_inc]]
        else:
            inc = 0
        if inc is None:
            inc = 0

        return inc

    def __call__(self, *keys):
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_vertexs(self):
        if hasattr(self, '__num_vertexs__'):
            return self.__num_vertexs__
        if self.x is not None:
            return self.x.size(self.__cat_dim__('x', self.x))
        if self.boundary_index is not None:
            return int(self.boundary_index[1, :].max()) + 1
        assert self.upper_index is None and self.lower_index is None
        return None

    @num_vertexs.setter
    def num_vertexs(self, num_vertexs):
        self.__num_vertexs__ = num_vertexs

    @property
    def num_vertexs_up(self):
        if hasattr(self, '__num_vertexs_up__'):
            return self.__num_vertexs_up__
        elif self.shared_coboundaries is not None:
            assert self.upper_index is not None
            return int(self.shared_coboundaries.max()) + 1
        assert self.upper_index is None
        return 0

    @num_vertexs_up.setter
    def num_vertexs_up(self, num_vertexs_up):
        self.__num_vertexs_up__ = num_vertexs_up

    @property
    def num_vertexs_down(self):
        if self.dim == 0:
            return None
        if hasattr(self, '__num_vertexs_down__'):
            return self.__num_vertexs_down__
        if self.lower_index is None:
            return 0
        raise ValueError('Cannot infer the number of vertexs in the cochain below.')

    @num_vertexs_down.setter
    def num_vertexs_down(self, num_vertexs_down):
        self.__num_vertexs_down__ = num_vertexs_down

    @property
    def num_features(self):
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, SparseTensor):
            try:
                return func(item)
            except AttributeError:
                return item
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict(
            {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
             for k, v in self.__dict__.items()})

    @property
    def mapping(self):
        return self.__mapping


class CochainBatch(Cochain):
    def __init__(self, dim, batch=None, ptr=None, **kwargs):
        super(CochainBatch, self).__init__(dim, **kwargs)

        for key, item in kwargs.items():
            if key == 'num_vertexs':
                self.__num_vertexs__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Cochain
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_vertexs_list__ = None
        self.__num_vertexs_down_list__ = None
        self.__num_vertexs_up_list__ = None
        self.__num_cochains__ = None

    @classmethod
    def from_cochain_list(cls, data_list, follow_batch=[]):
        keys = list(set.union(*[set(data.keys) for data in data_list]))
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls(data_list[0].dim)
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_cochains__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch']:
            batch[key] = []
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_vertexs_list = []
        num_vertexs_up_list = []
        num_vertexs_down_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                if item is not None:
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    if isinstance(item, Tensor) and item.dim() == 0:
                        item = item.unsqueeze(0)

                    batch[key].append(item)

                    size = 1
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()

                    slices[key].append(size + slices[key][-1])

                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(torch.full((size, ), i, dtype=torch.long, device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(torch.full((size, ), i, dtype=torch.long, device=device))

                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

            if hasattr(data, '__num_vertexs__'):
                num_vertexs_list.append(data.__num_vertexs__)
            else:
                num_vertexs_list.append(None)

            if hasattr(data, '__num_vertexs_up__'):
                num_vertexs_up_list.append(data.__num_vertexs_up__)
            else:
                num_vertexs_up_list.append(None)

            if hasattr(data, '__num_vertexs_down__'):
                num_vertexs_down_list.append(data.__num_vertexs_down__)
            else:
                num_vertexs_down_list.append(None)

            num_vertexs = data.num_vertexs
            if num_vertexs is not None:
                item = torch.full((num_vertexs, ), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_vertexs)

        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_vertexs_list__ = num_vertexs_list
        batch.__num_vertexs_up_list__ = num_vertexs_up_list
        batch.__num_vertexs_down_list__ = num_vertexs_down_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch.contiguous()

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(CochainBatch, self).__getitem__(idx)
        else:
            raise NotImplementedError

    @property
    def num_cochains(self) -> int:
        if self.__num_cochains__ is not None:
            return self.__num_cochains__
        return self.ptr.numel() + 1


class SSTree(object):
    def __init__(self, *cochains: Cochain, y: torch.Tensor = None, dimension: int = None):
        if len(cochains) == 0:
            raise ValueError('At least one cochain is required.')
        if dimension is None:
            dimension = len(cochains) - 1
        if len(cochains) < dimension + 1:
            raise ValueError(f'Not enough cochains passed, expected {dimension + 1}, received {len(cochains)}')

        self.dimension = dimension
        self.cochains = {i: cochains[i] for i in range(dimension + 1)}
        self.nodes = cochains[0]
        self.edges = cochains[1] if dimension >= 1 else None
        self.two_vertexs = cochains[2] if dimension >= 2 else None

        self.y = y

        self._consolidate()
        return

    def _consolidate(self):
        for dim in range(self.dimension + 1):
            cochain = self.cochains[dim]
            assert cochain.dim == dim
            if dim < self.dimension:
                upper_cochain = self.cochains[dim + 1]
                num_vertexs_up = upper_cochain.num_vertexs
                assert num_vertexs_up is not None
                if 'num_vertexs_up' in cochain:
                    assert cochain.num_vertexs_up == num_vertexs_up
                else:
                    cochain.num_vertexs_up = num_vertexs_up
            if dim > 0:
                lower_cochain = self.cochains[dim - 1]
                num_vertexs_down = lower_cochain.num_vertexs
                assert num_vertexs_down is not None
                if 'num_vertexs_down' in cochain:
                    assert cochain.num_vertexs_down == num_vertexs_down
                else:
                    cochain.num_vertexs_down = num_vertexs_down

    def to(self, device, **kwargs):
        for dim in range(self.dimension + 1):
            self.cochains[dim] = self.cochains[dim].to(device, **kwargs)
        if self.y is not None:
            self.y = self.y.to(device, **kwargs)
        return self

    def get_cochain_params(self,
                           dim: int,
                           max_dim: int = 2,
                           include_top_features=True,
                           include_down_features=True,
                           include_boundary_features=True) -> MPParams:

        if dim in self.cochains:
            vertexs = self.cochains[dim]
            x = vertexs.x

            upper_index, upper_features = None, None

            if vertexs.upper_index is not None and (dim + 1) in self.cochains:
                upper_index = vertexs.upper_index
                if self.cochains[dim + 1].x is not None and (dim < max_dim or include_top_features):
                    upper_features = torch.index_select(self.cochains[dim + 1].x, 0, self.cochains[dim].shared_coboundaries)

            lower_index, lower_features = None, None
            if include_down_features and vertexs.lower_index is not None:
                lower_index = vertexs.lower_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    lower_features = torch.index_select(self.cochains[dim - 1].x, 0, self.cochains[dim].shared_boundaries)

            boundary_index, boundary_features = None, None
            if include_boundary_features and vertexs.boundary_index is not None:
                boundary_index = vertexs.boundary_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    boundary_features = self.cochains[dim - 1].x

            inputs = MPParams(x,
                              upper_index,
                              lower_index,
                              up_attr=upper_features,
                              down_attr=lower_features,
                              boundary_attr=boundary_features,
                              boundary_index=boundary_index)
        else:
            raise NotImplementedError('Dim {} is not present or not yet supported.'.format(dim))
        return inputs

    def get_all_cochain_params(self,
                               max_dim: int = 2,
                               include_top_features=True,
                               include_down_features=True,
                               include_boundary_features=True) -> List[MPParams]:

        all_params = []
        return_dim = min(max_dim, self.dimension)
        for dim in range(return_dim + 1):
            all_params.append(
                self.get_cochain_params(dim,
                                        max_dim=max_dim,
                                        include_top_features=include_top_features,
                                        include_down_features=include_down_features,
                                        include_boundary_features=include_boundary_features))
        return all_params

    def get_labels(self, dim=None):
        if dim is None:
            y = self.y
        else:
            if dim in self.cochains:
                y = self.cochains[dim].y
            else:
                raise NotImplementedError('Dim {} is not present or not yet supported.'.format(dim))
        return y

    def set_xs(self, xs: List[Tensor]):
        assert (self.dimension + 1) >= len(xs)
        for i, x in enumerate(xs):
            self.cochains[i].x = x

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.keys


class SSTreeBatch(object):
    def __init__(self, dimension: int, num_data: int, x_list: List[Tensor], x_num_list: List[List], x_batch_list: List[Tensor],
                 x_index_list: List[Tensor], boundary_index_list: List[Tensor], upper_index_list: List[Tensor],
                 shared_coboundaries_list: List[Tensor], lower_index_list: List[Tensor], shared_boundaries_list: List[Tensor],
                 y: Tensor):

        self.dimension = dimension
        self.num_data = num_data
        self.y = y

        self.x_list = x_list
        self.x_num_list = x_num_list
        self.x_batch_list = x_batch_list
        self.x_index_list = x_index_list
        self.boundary_index_list = boundary_index_list
        self.upper_index_list = upper_index_list
        self.lower_index_list = lower_index_list
        self.shared_coboundaries_list = shared_coboundaries_list
        self.shared_boundaries_list = shared_boundaries_list

    @classmethod
    def from_sstree_list(cls, data_list: List[SSTree], follow_batch=[], max_dim: int = 2):

        # ts = time.time()
        dimension = max([data.dimension for data in data_list])
        dimension = min(dimension, max_dim)

        # y
        label_list = list()
        per_sstree_labels = True
        for tree in data_list:
            per_sstree_labels &= tree.y is not None
            if per_sstree_labels:
                label_list.append(tree.y)

        y = None if not per_sstree_labels else torch.cat(label_list, 0)

        # x, adj
        x_lists = [list() for _ in range(dimension + 1)]  # list of vertexs list
        x_num_lists = [list() for _ in range(dimension + 1)]  # # list of vertexs num list
        x_batch_lists = [list() for _ in range(dimension + 1)]  # list of vertexs batch idx
        x_index_lists = [list() for _ in range(dimension + 1)]
        boundary_index_lists = [list() for _ in range(dimension + 1)]
        upper_index_lists = [list() for _ in range(dimension + 1)]
        shared_coboundaries_lists = [list() for _ in range(dimension + 1)]
        lower_index_lists = [list() for _ in range(dimension + 1)]
        shared_boundaries_lists = [list() for _ in range(dimension + 1)]
        for i, tree in enumerate(data_list):
            for dim in range(dimension + 1):
                if dim in tree.cochains:
                    num_vertexs = tree.cochains[dim].num_vertexs
                    assert num_vertexs > 0
                    x_num_lists[dim].append(num_vertexs)
                    x_batch_lists[dim] += [i] * num_vertexs

                    idx1 = torch.arange(num_vertexs, dtype=torch.long)
                    idx0 = idx1.new_full(idx1.size(), i)
                    idx = torch.stack((idx0, idx1), dim=0)
                    x_index_lists[dim].append(idx)

                    if tree.cochains[dim].boundary_index is not None:
                        idx1 = tree.cochains[dim].boundary_index[1]
                        idx2 = tree.cochains[dim].boundary_index[0]
                        idx0 = idx1.new_full(idx1.size(), i)
                        idx = torch.stack((idx0, idx1, idx2), dim=0)
                        boundary_index_lists[dim].append(idx)

                    if tree.cochains[dim].upper_index is not None:
                        idx1 = tree.cochains[dim].upper_index[1]
                        idx2 = tree.cochains[dim].upper_index[0]
                        idx0 = idx1.new_full(idx1.size(), i)
                        idx = torch.stack((idx0, idx1, idx2), dim=0)
                        upper_index_lists[dim].append(idx)

                        shared_coboundaries_lists[dim].append(tree.cochains[dim].shared_coboundaries)

                    if tree.cochains[dim].lower_index is not None:
                        idx1 = tree.cochains[dim].lower_index[1]
                        idx2 = tree.cochains[dim].lower_index[0]
                        idx0 = idx1.new_full(idx1.size(), i)
                        idx = torch.stack((idx0, idx1, idx2), dim=0)
                        lower_index_lists[dim].append(idx)

                        shared_boundaries_lists[dim].append(tree.cochains[dim].shared_boundaries)

                    if tree.cochains[dim].x is not None:
                        x_lists[dim].append(tree.cochains[dim].x)
                else:
                    x_num_lists[dim].append(0)

        # x_num_list = [torch.tensor(_l, dtype=torch.long) for _l in x_num_lists]

        x_list = [torch.cat(_l, 0) if len(_l) > 0 else None for _l in x_lists]
        x_batch_list = [torch.tensor(_l, dtype=torch.long) if len(_l) > 0 else None for _l in x_batch_lists]
        x_index_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in x_index_lists]
        boundary_index_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in boundary_index_lists]
        upper_index_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in upper_index_lists]
        lower_index_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in lower_index_lists]
        shared_coboundaries_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in shared_coboundaries_lists]
        shared_boundaries_list = [torch.cat(_l, -1) if len(_l) > 0 else None for _l in shared_boundaries_lists]

        return cls(dimension=dimension,
                   num_data=len(data_list),
                   x_list=x_list,
                   x_num_list=x_num_lists,
                   x_batch_list=x_batch_list,
                   x_index_list=x_index_list,
                   boundary_index_list=boundary_index_list,
                   upper_index_list=upper_index_list,
                   shared_coboundaries_list=shared_coboundaries_list,
                   lower_index_list=lower_index_list,
                   shared_boundaries_list=shared_boundaries_list,
                   y=y)

    def to(self, device, **kwargs):
        self.y = None if self.y is None else self.y.to(device, **kwargs)

        # self.x_num_list = [_t.to(device, **kwargs) for _t in self.x_num_list]

        self.x_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.x_list]
        self.x_batch_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.x_batch_list]
        self.x_index_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.x_index_list]
        self.boundary_index_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.boundary_index_list]
        self.upper_index_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.upper_index_list]
        self.shared_coboundaries_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.shared_coboundaries_list]
        self.lower_index_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.lower_index_list]
        self.shared_boundaries_list = [_t if _t is None else _t.to(device, **kwargs) for _t in self.shared_boundaries_list]

        return self

    def get_all_cochain_params(self, max_dim: int = 2) -> List[MPParams]:
        all_params = []
        return_dim = min(max_dim, self.dimension)
        for dim in range(return_dim + 1):
            x = self.x_list[dim]
            x_num = self.x_num_list[dim]
            x_batch = self.x_batch_list[dim]
            x_index = self.x_index_list[dim]
            batch_size = self.num_data

            boundary_index = self.boundary_index_list[dim]
            upper_index = self.upper_index_list[dim]
            shared_coboundaries = self.shared_coboundaries_list[dim]
            lower_index = self.lower_index_list[dim]
            shared_boundaries = self.shared_boundaries_list[dim]

            param = MPParams(batch_size, x, x_num, x_batch, x_index, boundary_index, upper_index, shared_coboundaries,
                             lower_index, shared_boundaries)
            all_params.append(param)

        return all_params
