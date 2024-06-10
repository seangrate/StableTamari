from collections import Counter
import itertools as it

from orders import position_compare, tamari_compare
from posets import LowerOrderIdeal


def main():
    a = [6, 7, 0, 2, 1]
    b = [10, 9, 0, 4, 3]
    c = [8, 7, 2, 3, 1]

    # Example 1 (position order)
    # should be diagonal and 1 <= 3, 2 <= 3, and 4 <= 5
    for (i,j) in sorted(it.combinations_with_replacement(range(0,5), 2)):
        if position_compare(i, j, a):
            print(f'{i+1} <= {j+1}')
    
    # Example 2 (stable Tamari order)
    # should be a <= b and a !<= c
    print(f'{a=} <= {b=}: {tamari_compare(a, b)}')
    print(f'{a=} <= {c=}: {tamari_compare(a, c)}')    


if __name__ == '__main__':
    main()