from collections import Counter, OrderedDict, defaultdict
import functools as ft
import itertools as it
from typing import List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from orders import tamari_compare
from utils import TropicalMatrix, format_monomial


class LowerOrderIdeal:
    def __init__(self, root: Sequence[int], mobius_lower: List[int]=None):
        self.root = root

        if mobius_lower is None:
            mobius_lower = [0] * len(self.root)
        
        # Format is:
        # `key` = sequence as tuple
        # `value` = {"covered_by": set(), "covers": set()}
        self.covering_relations = defaultdict(lambda: {'covered_by': set(), 'covers': set()})

        # Construct covering relations
        todo = set([self.root])
        while len(todo) > 0:
            new_todo = set()
            for a in todo:
                for k in self.D(a):
                    b = self.partial(a, k)
                    self.covering_relations[a]["covers"].add(b)
                    self.covering_relations[b]["covered_by"].add(a)
                    new_todo.add(b)
            todo = new_todo
        self.nodes = list(self.covering_relations.keys())

    def covers(self, upper_node: Sequence[int], lower_node: Sequence[int]) -> bool:
        """Checks if the upper node covers the lower node."""
        return lower_node in self.covering_relations[upper_node]['covers']
    
    def covered_by(self, lower_node: Sequence[int], upper_node: Sequence[int]) -> bool:
        """Checks if the lower node is covered by the upper node."""
        return upper_node in self.covering_relations[lower_node]['covered_by']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.root})'
    
    def __len__(self):
        return len(self.nodes)

    @staticmethod
    def h(a):
        ret = []
        for i in range(len(a)):
            j = i
            while j > 0:
                if a[j-1] <= a[i]:
                    break
                j -= 1
            ret.append(j)
        return ret

    @staticmethod
    def D(a):
        h_vec = LowerOrderIdeal.h(a)
        ret = []
        for i in range(len(a)):
            x = 0 if h_vec[i] == 0 else a[h_vec[i]-1]
            y = a[i]
            if x < y:
                ret.append(i)
        return ret

    @staticmethod
    def partial(a, k):
        h_vec = LowerOrderIdeal.h(a)
        return tuple([a[i] - 1 if h_vec[k] <= i <= k else a[i] for i in range(len(a))])
        
    @ft.cached_property
    def order_matrix(self) -> np.ndarray:
        # the value in entry (i,j) stores if node i <= node j in the stable Tamari order
        # with how itertools creates the product, only need to regard this as an upper triangular matrix
        # since the stable Tamari order compares coordinates
        return np.array([[tamari_compare(a, b) for b in self.nodes] for a in self.nodes], dtype=bool)
    
    @ft.cached_property
    def height_matrix(self) -> np.ndarray:
        # prep order matrix for tropical arithmetic
        # if entry (i,j) is inf, then the height is not defined
        # otherwise, it is the length of a maximum chain in the poset
        comparable_indices = np.where(self.order_matrix != 0)
        tropical_order_matrix = np.full(self.order_matrix.shape, -np.inf)
        tropical_order_matrix[comparable_indices] = 1
        np.fill_diagonal(tropical_order_matrix, 0)
        tropical_order_matrix = TropicalMatrix(tropical_order_matrix)
        return (tropical_order_matrix**sum(self.root))._data

    def children(self, parent) -> List[Sequence[int]]:
        parent_idx = self.nodes.index(parent)
        return [node for child_idx, node in enumerate(self.nodes) if self.order_matrix[child_idx, parent_idx]]

    def hasse_diagram(self, nodes_as_boxes: bool=False,
                      show: bool=True, figsize=(8,8), **kwargs):
        # make the Hasse diagram
        G = nx.DiGraph()
        nodes = [node for node in self.nodes]
        G.add_nodes_from([(node, {'height': -self.height(node, self.root)}) for node in nodes])
        # G.add_nodes_from([(node, {'height': self.height(tuple(np.zeros_like(node)), node)}) for node in nodes])
        for i, node in enumerate(G.nodes):
            G.add_edges_from([(node, other_node) for j, other_node in enumerate(G.nodes) if i != j and self.height_matrix[i, j] == 1])
        
        # plot the Hasse diagram
        ax = kwargs.get('ax')
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            kwargs['ax'] = ax
        pos = nx.multipartite_layout(G, subset_key='height', align='horizontal')
        labels = {node: self.as_boxes(node) for node in nodes} if nodes_as_boxes else {node: str(node) for node in nodes}
        for u, data in G.nodes(data=True):
            x, y = pos[u]
            bbox_props = dict(edgecolor='black', facecolor='white', boxstyle='round,pad=1.0')
            ax.text(x, y, labels[u], ha='center', va='center', bbox=bbox_props, fontdict={'family': 'monospace'})
        nx.draw(G, pos, with_labels=False, labels=labels, **kwargs)
        if show:
            plt.show()
        return G, ax

    def as_boxes(self, vector: Sequence[int]) -> str:
        max_row_length = max(self.root)
        # ■□○
        # use a non-breaking space character to ensure whitespace is preserved
        nbsp = '\u00A0'
        return '\n'.join(f'{(max_row_length - b + k)*nbsp}{(b-k)*"■"}|{k*"□"}{(max_row_length - k)*nbsp}' for (b, k) in zip(self.root, vector))

    def _add_edges(self,graph, node, parent=None):
        if node is not None:
            if parent is not None:
                graph.add_edge(node, parent)
            for child in node.children:
                self._add_edges(graph, child, parent=node)

    def height(self, lower, upper) -> int:
        # length of a maximal chain between a and b
        lower_idx, upper_idx = self.nodes.index(lower), self.nodes.index(upper)
        if not self.order_matrix[lower_idx, upper_idx]: 
            raise ValueError(f'{lower} is not less than {upper} in the stable Tamari order.')
        return int(self.height_matrix[lower_idx, upper_idx])
    
    def interval(self, bottom_node, top_node) -> List[Sequence[int]]:
        """Computes the interval [a,b] in the lower order ideal.
        
        Given nodes a and b in the lower order ideal, the interval [a,b]
        consists of all nodes x in the lower order ideal such that a <= x <= b.
        
        Parameters
        ----------
        bottom_node : Sequence[int]
            The minimum node a of the interval.
        top_node : Sequence[int]
            The maximum node b of the interval.

        Returns
         : List[Sequence[int]]
            A list of the nodes in the interval [a,b].
        """
        return [x for x in self.nodes if tamari_compare(bottom_node, x) and tamari_compare(x, top_node)]


    @ft.cached_property
    def f_polynomial_counts(self) -> OrderedDict:
        top_idx = self.nodes.index(self.root)
        finite_heights = self.height_matrix[:, top_idx].flatten()
        finite_heights = finite_heights[np.isfinite(finite_heights)].astype(int)
        return OrderedDict(sorted(Counter(finite_heights).items(), reverse=True))

    @ft.cached_property
    def f_polynomial(self) -> str:
        return ' + '.join(format_monomial(count, height, 't') for height, count in self.f_polynomial_counts.items())

    @ft.cached_property
    def F_polynomial_counts(self) -> OrderedDict:
        finite_heights = self.height_matrix.flatten()
        finite_heights = finite_heights[np.isfinite(finite_heights)].astype(int)
        return OrderedDict(sorted(Counter(finite_heights).items(), reverse=True))
    
    @ft.cached_property
    def F_polynomial(self) -> str:
        return ' + '.join(format_monomial(count, height, 't') for height, count in self.F_polynomial_counts.items())