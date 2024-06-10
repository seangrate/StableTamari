import itertools as it

import numpy as np

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
        if not isinstance(n, int):
            raise TypeError(f'{n} must be an integer.')
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