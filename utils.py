from collections import defaultdict
import itertools as it
from typing import List, Sequence, Set

import math
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# for compatibility with Sage
try:
    from sage.rings.integer import Integer
except ImportError as e:
    pass


class TropicalMatrix:
    def __init__(self, data: np.ndarray):
        self._data = data

    def __repr__(self):
        return self._data.__repr__()
    
    def __str__(self):
        return self._data.__str__()

    def __getitem__(self, idx):
        submatrix = TropicalMatrix(self._data[idx])
        if not submatrix.shape:
            return submatrix._data.item()
        return TropicalMatrix(submatrix)

    def __add__(self, other):
        if self.shape != other.shape:
            raise ValueError(f'operands could not be broadcast together with shapes {self.shape} {other.shape}')  
        return TropicalMatrix(np.maximum(self._data, other._data))

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        prod_matrix = np.empty(shape=(self.shape[0], other.shape[1]))
        for i, j in it.product(range(self.shape[0]), range(other.shape[1])):
            prod_matrix[i, j] = np.max(self._data[i, :] + other._data[:, j])
        return TropicalMatrix(prod_matrix)

    def __rmatmul__(self, other):
        return other.__matmult__(self)
    
    def __pow__(self, n: int):
        # if not isinstance(n, int):
        #     # Sage shenanigans
        #     try:
        #         if isinstance(n, Integer):
        #             n = int(n)
        #     except NameError as e:
        #         raise TypeError(f'{n} must be an integer.')
        if n < 0:
            raise ValueError(f'{n} must be nonnegative.')
        # NOTE: Taken from NumPy's matrix power function for convenenience.
        # Use binary decomposition to reduce the number of matrix multiplications.
        # Here, we iterate over the bits of n, from LSB to MSB, raise `a` to
        # increasing powers of 2, and multiply into the result as needed.
        z = result = None
        while n > 0:
            z = self if z is None else z @ z
            n, bit = divmod(n, 2)
            if bit:
                result = z if result is None else result @ z
        return result

    @property
    def shape(self):
        return self._data.shape
    

    ######### THIS IS SUPPOSED TO DO STUFF ##########    
    # # use tropical arithmetic to figure out if there is a walk of length <= r
    # # by computing tropical matrix powers
    # adj_matrix = nx.adjacency_matrix(self).toarray().astype(float)
    # adj_matrix[adj_matrix == 0] = np.inf
    # np.fill_diagonal(adj_matrix, 0)
    # walks_matrix = TropicalMatrix(adj_matrix)**r
    # origin_idx = list(self.nodes()).index((0,0))
    # return {(0,0)} | {vertex for idx, vertex in enumerate(self.nodes()) if walks_matrix[origin_idx, idx] < np.inf}


def format_monomial(coeff: int, exponent: int, variable: str='t') -> str:
    # coefficient is never zero a fiat
    # if exponent == 0:
    #     return f'{coeff}' 
    sign_str = '' if coeff >= 0 else '-'
    coeff_str = '' if abs(coeff) == 1 else str(abs(coeff))
    variable_str = '' if exponent == 0 else variable
    exponent_str = '' if exponent <= 1 else f'^{exponent}'
    # if coeff == 1:
    #     if exponent == 0:
    #         return f'{coeff}'
    #     elif exponent == 1:
    #         return f'{variable}'
    #     else:
    #         return f'{variable}^{exponent}'
    # elif coeff == -1:
    #     if exponent == 0:
    #         return f'{coeff}'
    #     elif exponent == 1:
    #         return f'-{variable}'
    #     else:
    #         return f'-{variable}^{exponent}'
    # else:
    #     if exponent == 0:
    #         return f'{coeff}'
    #     elif exponent == 1:
    #         return f'{coeff}{variable}'
    #     else:
    #         return f'{coeff}{variable}^{exponent}'
    return f'{sign_str}{coeff_str}{variable_str}{exponent_str}'
            

def coplanar_faces(hull):
    coplanar_mask = np.zeros((len(hull.simplices), len(hull.simplices)), dtype=bool)
    for i, simplex in enumerate(hull.simplices):
        for j, other_simplex in enumerate(hull.simplices):
            if i != j:
                if j in hull.neighbors[i]:
                    common_indices = [idx for idx in simplex if idx in other_simplex]
                    simplex_diff_idx = list(set(simplex) - set(other_simplex))[0]
                    other_simplex_diff_idx = list(set(other_simplex) - set(simplex))[0]
                    defining_cross = np.cross(hull.points[common_indices[1]] - hull.points[common_indices[0]], hull.points[simplex_diff_idx] - hull.points[common_indices[0]])
                    coplanar_mask[i, j] = math.isclose(defining_cross.dot(hull.points[other_simplex_diff_idx] - hull.points[common_indices[0]]), 0)
                else:
                    coplanar_mask[i, j] = False
            else:
                coplanar_mask[i, j] = True
    return coplanar_mask


def facet_points(hull, coplanar_mask):
    facet_points = defaultdict(set)
    visited_simplices = set()
    for i, simplex in enumerate(hull.simplices):
        if i not in visited_simplices:
            visited_simplices.add(i)
            facet_points[i].update(simplex)
            for j, other_simplex in enumerate(hull.simplices):
                if coplanar_mask[i, j]:
                    visited_simplices.add(j)
                    facet_points[i].update(other_simplex)
    return list(map(list, facet_points.values()))


def clockwise_around_center(point, centroid, defining_cross):
    """
    
    Taken from https://stackoverflow.com/a/74413211
    """
    # make arctan2 function that returns a value from [0, 2 pi) instead of [-pi, pi)
    arctan2 = lambda s, c: angle if (angle := np.arctan2(s, c)) >= 0 else 2 * np.pi + angle
    
    diff = point - centroid
    rcos = np.dot(diff, centroid)
    rsin = np.dot(defining_cross, np.cross(diff, centroid))
    return arctan2(rsin, rcos).item()


def plot_convex_hull(hull):
    dim = hull.points.shape[1]
    if dim not in {2,3}:
        raise ValueError(f'Plotting only supported for 2D and 3D convex hulls, got {dim}D.')
    points = hull.points

    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot(111)
        hull_vertices = points[hull.vertices]
        hull_vertices = np.append(hull_vertices, [hull_vertices[0]], axis=0)
        ax.plot(hull_vertices[:, 0], hull_vertices[:, 1], 'r--')
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
        facets = facet_points(hull, coplanar_faces(hull))
        for facet in facets:
            # reorder to be clockwise around the centroid
            centroid = np.mean(points[facet], axis=0)
            defining_cross = np.cross(points[facet[1]] - points[facet[0]], points[facet[2]] - points[facet[0]])
            sorted_facet = sorted(facet, key=lambda idx: clockwise_around_center(points[idx], centroid, defining_cross))
            cyclic_facet = np.append(sorted_facet, sorted_facet[0])
            # plot the edges
            ax.plot(points[cyclic_facet, 0], points[cyclic_facet, 1], points[cyclic_facet, 2], 'k-')
            
            # plot the facets
            vertices = [points[facet_idx] for facet_idx in cyclic_facet]
            # vertices = [points[simplex[0]], points[simplex[1]], points[simplex[2]], points[simplex[0]]]
            mpl_facet = Poly3DCollection([vertices], alpha=0.25, facecolor='cyan')
            ax.add_collection3d(mpl_facet)
        ax.set_zlim(-0.1, dim + .1)
    ax.set_xlim(-0.1, dim + .1)
    ax.set_ylim(-0.1, dim + .1)
    return ax
    

def find_peaks(sequence: Sequence[int], zero_indexed: bool=False) -> List[int]:
    """Finds the indices of the peaks in a sequence.

    Taken from https://stackoverflow.com/a/74556122
    """
    peak_indices = ((np.diff(np.sign(np.diff(sequence))) < 0).nonzero()[0] + 1).tolist()
    if sequence[0] > sequence[1]:
        peak_indices = [0] + peak_indices
    if sequence[-1] > sequence[-2]:
        peak_indices = peak_indices + [len(sequence) - 1]
    return peak_indices if zero_indexed else [idx+1 for idx in peak_indices]


def is_unimodal(sequence: Sequence[int]) -> bool:
    # # calculate consecutive, pairwise differences and see if sign changed in difference (increasing to decreasing or vice versa)
    # # sign can only change at most one time
    # first_diffs = [a-b for (a, b) in zip(sequence[:-1], sequence[1:]) if a-b != 0]  # discard 0 change
    # sign_flips = [1 if a*b < 0 else 0 for (a, b) in zip(first_diffs[:-1], first_diffs[1:])]
    # return sum(sign_flips) <= 1
    return len(find_peaks(sequence)) <= 1


def is_log_concave(sequence: Sequence[int]) -> bool:
    return all(b**2 >= a*c for (a, b, c) in zip(sequence[:-2], sequence[1:-1], sequence[2:]))


def affine_transform(sequence):
    """Computes the affine transformation between two polytopes defined by unimoal permutations.
    
    Notes
    -----
    Assumes that the sequence is a unimodal permutation.
    Right now, this also assume that this is the transformation from P(sigma) to P((1,2,...,n)).
    """
    if not is_unimodal(sequence):
        raise ValueError(f'{sequence} must be unimodal.')
    peak_idx = find_peaks(sequence, zero_indexed=True)[0]

    transition_matrix = np.zeros((len(sequence), len(sequence)), dtype=int)
    for j, (start_row_idx, end_row_idx) in enumerate(zip(sequence[:-1], sequence[1:])):
        start_row_idx, end_row_idx = sorted((start_row_idx, end_row_idx))
        fill_value = 1 if j < peak_idx else -1
        if j >= peak_idx:
            j = j+1
        transition_matrix[start_row_idx-1:end_row_idx-1, j] = fill_value
    if peak_idx != 0:
        transition_matrix[-1, peak_idx-1] = 1
    transition_matrix[-1, peak_idx] = -1

    shift_vector = np.hstack([np.zeros((peak_idx,), dtype=int), np.array(sequence)[peak_idx:]]).reshape(-1, 1)

    return transition_matrix, shift_vector