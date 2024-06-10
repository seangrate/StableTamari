import itertools as it
from typing import Sequence


def is_partition(seq: Sequence[int]):
    if any(k < 0 for k in seq):
        return False
    return sorted(seq) == list(seq)


def position_compare(i: int, j: int, a: Sequence[int]) -> bool:
    """Computes a <= b in the position order induced by a.
    
    Given two vectors of nonnegative integers a,b, we say a <= b in the position
    order if i <= j and a_i, a_(i+1), ..., a_(j-1) > a_j.

    Parameters
    ----------
    i : int
        a nonnegative integer
    j : int
        a nonnegative integer
    a : Tuple[int]
        a vector of length n of nonnegative integers

    Returns
    -------
     : bool
        whether i <= j in the position order induced by a
    """
    if i >= len(a):
        raise ValueError(f'{i=} must satisfy 0 <= i <= {len(a)-1}')
    if j >= len(a):
        raise ValueError(f'{j=} must satisfy 0 <= j <= {len(a)-1}')
    if i > j:
        return False
    return all(a[k] > a[j] for k in range(i, j))


def tamari_compare(a: Sequence[int], b: Sequence[int]) -> bool:
    """Computes a <= b in the stable Tamari order.

    Given two vectors of nonnegative integers a,b, we say a <= b in the stable 
    Tamari order if a <= b coordinate-wise and a_i - a_j <= b_i - b_j for all 
    i <= j (in the position order).

    Note that a and b must have the same size.
    
    Parameters
    ----------
    a : Tuple[int]
        a vector of nonnegative integers
    b : Tuple[int]
        a vector of nonnegative integers.

    Returns
    -------
     : bool
        whether a <= b in the stable Tamari order
    """
    if len(a) != len(b):
        raise ValueError(f'The length of {a} and {b} must be the same.')
    # # Theorem 1 of proposal paper
    # if is_partition(a) and is_partition(b):
    #     return is_partition([bi - ai for (ai, bi) in zip(a, b)])
    # normal stable Tamari comparison method
    if not all(ai <= bi for (ai, bi) in zip(a, b)):
        return False
    return all(a[i] - a[j] <= b[i] - b[j] for (i, j) in it.combinations(range(len(a)), 2) if position_compare(i, j, a))
