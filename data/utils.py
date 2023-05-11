import numpy as np
import torch
import gudhi as gd
import itertools
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx

from tqdm import tqdm
from data.sstree import Cochain, SSTree
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from data.parallel import ProgressParallel
from joblib import delayed


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    st = gd.SimplexTree()

    for v in range(size):
        st.insert([v])

    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    boundaries = itertools.combinations(simplex, len(simplex) - 1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    sstree_dim = simplex_tree.dimension()

    id_maps = [{} for _ in range(sstree_dim + 1)]
    simplex_tables = [[] for _ in range(sstree_dim + 1)]
    boundaries_tables = [[] for _ in range(sstree_dim + 1)]

    simplex_tables[0] = [[v] for v in range(size)]
    id_maps[0] = {tuple([v]): v for v in range(size)}

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        next_id = len(simplex_tables[dim])
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, sstree_dim: int):
    boundaries = [{} for _ in range(sstree_dim + 2)]
    coboundaries = [{} for _ in range(sstree_dim + 2)]
    boundaries_tables = [[] for _ in range(sstree_dim + 1)]

    for simplex, _ in simplex_tree.get_simplices():
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        if simplex_dim > 0:
            boundaries_ids = [id_maps[simplex_dim - 1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], sstree_dim: int, include_down_adj: bool):
    def initialise_structure():
        return [[] for _ in range(sstree_dim + 1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    for dim in range(sstree_dim + 1):
        for simplex, id in id_maps[dim].items():
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])
                    all_shared_coboundaries[dim - 1].extend([id, id])

            if include_down_adj and dim < sstree_dim and simplex in coboundaries[dim]:
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])
                    all_shared_boundaries[dim + 1].extend([id, id])

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, vertex_tables, init_method: str) -> List:
    features = [vx]
    for dim in range(1, len(vertex_tables)):
        aux_1 = []
        aux_0 = []
        for c, vertex in enumerate(vertex_tables[dim]):
            aux_1 += [c for _ in range(len(vertex))]
            aux_0 += vertex
        node_vertex_index = torch.LongTensor([aux_0, aux_1])
        in_features = vx.index_select(0, node_vertex_index[0])
        features.append(scatter(in_features, node_vertex_index[1], dim=0, dim_size=len(vertex_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    v_y, sstree_y = None, None
    if y is None:
        return v_y, sstree_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        sstree_y = y
    else:
        assert y_shape[0] == size
        v_y = y

    return v_y, sstree_y


def generate_cochain(dim,
                     x,
                     all_upper_index,
                     all_lower_index,
                     all_shared_boundaries,
                     all_shared_coboundaries,
                     vertex_tables,
                     boundaries_tables,
                     sstree_dim,
                     y=None):
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_vertexs_down = len(vertex_tables[dim - 1]) if dim > 0 else None
    num_vertexs_up = len(vertex_tables[dim + 1]) if dim < sstree_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t() if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t() if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                           if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                         if len(all_shared_boundaries[dim]) > 0 else None)

    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, vertex in enumerate(boundaries_tables[dim]):
            for boundary in vertex:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)

    if num_vertexs_down is None:
        assert shared_boundaries is None
    if num_vertexs_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_vertexs_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_vertexs_down >= shared_boundaries.max() + 1

    return Cochain(dim=dim,
                   x=x,
                   upper_index=up_index,
                   lower_index=down_index,
                   shared_coboundaries=shared_coboundaries,
                   shared_boundaries=shared_boundaries,
                   y=y,
                   num_vertexs_down=num_vertexs_down,
                   num_vertexs_up=num_vertexs_up,
                   boundary_index=boundary_index)


def compute_clique_sstree_with_gudhi(x: Tensor,
                                     edge_index: Adj,
                                     size: int,
                                     expansion_dim: int = 2,
                                     y: Tensor = None,
                                     include_down_adj=True,
                                     init_method: str = 'sum') -> SSTree:

    assert x is not None
    assert isinstance(edge_index, Tensor)

    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(expansion_dim)
    sstree_dim = simplex_tree.dimension()

    simplex_tables, id_maps = build_tables(simplex_tree, size)

    boundaries_tables, boundaries, co_boundaries = (extract_boundaries_and_coboundaries_from_simplex_tree(
        simplex_tree, id_maps, sstree_dim))

    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps, sstree_dim,
                                                                             include_down_adj)

    xs = construct_features(x, simplex_tables, init_method)

    v_y, sstree_y = extract_labels(y, size)

    cochains = []
    for i in range(sstree_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i,
                                   xs[i],
                                   upper_idx,
                                   lower_idx,
                                   shared_boundaries,
                                   shared_coboundaries,
                                   simplex_tables,
                                   boundaries_tables,
                                   sstree_dim=sstree_dim,
                                   y=y)
        cochains.append(cochain)

    return SSTree(*cochains, y=sstree_y, dimension=sstree_dim)


def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True, init_method: str = 'sum'):
    dimension = -1
    sstrees = []
    num_features = [None for _ in range(expansion_dim + 1)]

    for data in tqdm(dataset):
        sstree = compute_clique_sstree_with_gudhi(data.x,
                                                  data.edge_index,
                                                  data.num_nodes,
                                                  expansion_dim=expansion_dim,
                                                  y=data.y,
                                                  include_down_adj=include_down_adj,
                                                  init_method=init_method)
        if sstree.dimension > dimension:
            dimension = sstree.dimension
        for dim in range(sstree.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = sstree.cochains[dim].num_features
            else:
                assert num_features[dim] == sstree.cochains[dim].num_features
        sstrees.append(sstree)

    return sstrees, dimension, num_features[:dimension + 1]


def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)

    rings = set()
    sorted_rings = set()
    for k in range(3, max_k + 1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True, generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):

    vertex_tables, id_maps = build_tables(simplex_tree, size)

    rings = get_rings(edge_index, max_k=max_k)

    if len(rings) > 0:
        id_maps += [{}]
        vertex_tables += [[]]
        assert len(vertex_tables) == 3, vertex_tables
        for vertex in rings:
            next_id = len(vertex_tables[2])
            id_maps[2][vertex] = next_id
            vertex_tables[2].append(list(vertex))

    return vertex_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
        simplex_tree, id_maps, simplex_tree.dimension())

    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for vertex in id_maps[2]:
            vertex_boundaries = get_ring_boundaries(vertex)
            boundaries[2][vertex] = list()
            boundaries_tables[2].append([])
            for boundary in vertex_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][vertex].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(vertex)
                boundaries_tables[2][-1].append(id_maps[1][boundary])

    return boundaries_tables, boundaries, coboundaries


def compute_ring_2sstree(x: Union[Tensor, np.ndarray],
                         edge_index: Union[Tensor, np.ndarray],
                         edge_attr: Optional[Union[Tensor, np.ndarray]],
                         size: int,
                         y: Optional[Union[Tensor, np.ndarray]] = None,
                         max_k: int = 7,
                         include_down_adj=True,
                         init_method: str = 'sum',
                         init_edges=True,
                         init_rings=False) -> SSTree:

    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0

    vertex_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    assert len(id_maps) <= 3
    sstree_dim = len(id_maps) - 1

    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)

    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps, sstree_dim,
                                                                             include_down_adj)

    xs = [x, None, None]
    constructed_features = construct_features(x, vertex_tables, init_method)
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]

    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        else:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            max_id = max(ex.keys())
            edge_feats = []
            assert len(vertex_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)

    v_y, sstree_y = extract_labels(y, size)

    cochains = []
    for i in range(sstree_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i,
                                   xs[i],
                                   upper_idx,
                                   lower_idx,
                                   shared_boundaries,
                                   shared_coboundaries,
                                   vertex_tables,
                                   boundaries_tables,
                                   sstree_dim=sstree_dim,
                                   y=y)
        cochains.append(cochain)

    return SSTree(*cochains, y=sstree_y, dimension=sstree_dim)


def convert_graph_dataset_with_rings(dataset,
                                     max_ring_size=7,
                                     include_down_adj=False,
                                     init_method: str = 'sum',
                                     init_edges=True,
                                     init_rings=False,
                                     n_jobs=1):
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    sstrees = parallel(
        delayed(compute_ring_2sstree)(maybe_convert_to_numpy(data.x),
                                      maybe_convert_to_numpy(data.edge_index),
                                      maybe_convert_to_numpy(data.edge_attr),
                                      data.num_nodes,
                                      y=maybe_convert_to_numpy(data.y),
                                      max_k=max_ring_size,
                                      include_down_adj=include_down_adj,
                                      init_method=init_method,
                                      init_edges=init_edges,
                                      init_rings=init_rings) for data in dataset)

    for c, sstree in enumerate(sstrees):

        if sstree.dimension > dimension:
            dimension = sstree.dimension
        for dim in range(sstree.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = sstree.cochains[dim].num_features
            else:
                assert num_features[dim] == sstree.cochains[dim].num_features

        graph = dataset[c]
        if sstree.y is None:
            assert graph.y is None
        else:
            mask = ~torch.isnan(sstree.y)
            assert torch.equal(sstree.y[mask], graph.y[mask])
        assert torch.equal(sstree.cochains[0].x, graph.x)
        if sstree.dimension >= 1:
            assert sstree.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return sstrees, dimension, num_features[:dimension + 1]
