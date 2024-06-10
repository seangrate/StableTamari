from collections import Counter, OrderedDict
import functools as ft
import itertools as it
from typing import List, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from orders import tamari_compare
from utils import TropicalMatrix


class LowerOrderIdeal:
    def __init__(self, b: Sequence[int]):
        self.root = b

    def __repr__(self):
        return f'{self.__class__.__name__}({self.root})'
    
    def __len__(self):
        return len(self.nodes)

    @ft.cached_property
    def nodes(self) -> List[Sequence[int]]:
        return [vector for vector in list(it.product(*(range(k+1) for k in self.root))) if tamari_compare(vector, self.root)]
    
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

    def hasse_diagram(self):
        G = nx.DiGraph()
        nodes = [node for node in self.nodes]
        G.add_nodes_from([(node, {'height': sum(node)}) for node in nodes])
        for i, node in enumerate(G.nodes):
            G.add_edges_from([(node, other_node) for j, other_node in enumerate(G.nodes) if i != j and self.height_matrix[i, j] == 1])
        nx.draw(G, nx.multipartite_layout(G, subset_key='height', align='horizontal'), 
                with_labels=True, labels={node: str(node) for node in nodes})
        plt.show()

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

    @ft.cached_property
    def f_polynomial_counts(self) -> OrderedDict:
        top_idx = self.nodes.index(self.root)
        finite_heights = self.height_matrix[:, top_idx].flatten()
        finite_heights = finite_heights[np.isfinite(finite_heights)].astype(int)
        return OrderedDict(sorted(Counter(finite_heights).items(), reverse=True))

    @ft.cached_property
    def f_polynomial(self) -> str:
        return ' + '.join(self._format_monomial(count, height, 't') for height, count in self.f_polynomial_counts.items())

    @ft.cached_property
    def F_polynomial_counts(self) -> OrderedDict:
        finite_heights = self.height_matrix.flatten()
        finite_heights = finite_heights[np.isfinite(finite_heights)].astype(int)
        return OrderedDict(sorted(Counter(finite_heights).items(), reverse=True))
    
    @ft.cached_property
    def F_polynomial(self) -> str:
        return ' + '.join(self._format_monomial(count, height, 't') for height, count in self.F_polynomial_counts.items())
    
    def _format_monomial(self, coeff: int, exponent: int, variable: str='t') -> str:
        # coefficient is never zero a fiat
        # if exponent == 0:
        #     return f'{coeff}'        
        if coeff == 1:
            if exponent == 0:
                return f'{coeff}'
            elif exponent == 1:
                return f'{variable}'
            else:
                return f'{variable}^{exponent}'
        else:
            if exponent == 0:
                return f'{coeff}'
            elif exponent == 1:
                return f'{coeff}{variable}'
            else:
                return f'{coeff}{variable}^{exponent}'
            
    @staticmethod
    def is_unimodal(sequence: Sequence[int]) -> bool:
        # calculate consecutive, pairwise differences and see if sign changed in difference (increasing to decreasing or vice versa)
        # sign can only change at most one time        
        first_diffs = [a-b for (a, b) in zip(sequence[:-1], sequence[1:]) if a-b != 0]  # discard 0 change
        sign_flips = [1 if a*b < 0 else 0 for (a, b) in zip(first_diffs[:-1], first_diffs[1:])]
        return sum(sign_flips) <= 1