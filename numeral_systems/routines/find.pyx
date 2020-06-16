import numpy as np
from inspect import isfunction


def find(iterable, cond, axis=0):
    """Find the indices at which a condition is true or a value/vector occurs. The iterable can be a list or 1d/2d array, while the condition can be a value, list, or 1d array.
    axis 0: find rows in which values from cond occur. length of cond must match number of columns if vector.
    axis 1: find rows in which values from cond occur. length of cond must match number of rows if vector. """
    if isinstance(iterable, list) and callable(cond):
        return [i for i, val in enumerate(iterable) if cond(val)]
    elif isinstance(iterable, list) and not callable(cond):
        return [i for i, val in enumerate(iterable) if val == cond]
    elif isinstance(iterable, np.ndarray):
        if iterable.ndim == 1:
            return find(list(iterable), cond)
        else:
            if callable(cond):
                return [index for index, elem in np.ndenumerate(iterable) if cond(elem)]
            else:
                if (isinstance(cond, np.ndarray) and cond.ndim == 1) or isinstance(cond, list):
                    res = []
                    for i in range(iterable.shape[axis]):
                        for j in range(iterable.shape[abs(axis - 1)]): #only supports 2d arrays for now
                            if axis == 1:
                                if iterable[j, i] == cond[i]:
                                    res.append(j)
                                    break
										
                            elif axis == 0:
                                if iterable[i, j] == cond[i]:
                                    res.append(j)
                                    break
                    return res
					
                elif cond.ndim == 0:
                    return 	[index for index, elem in np.ndenumerate(iterable) if elem == cond]
                else:
                    raise NotImplementedError


def find_diff(X, Y):	
    #to match matlab implementation
    return sorted(set(X) - set(Y))

def find_unique(X):
    return sorted(set(X))


if __name__ == "__main__":
    #print(find(compute_f_i_w_numerator(100, 3, [i for i in range(1, 101)], [1, 74, 75], 2.2810, 0.31), ))
    print(find(np.arange(18).reshape(3, 6), [0, 13, 8, 3, 4, 11], axis=1))
    print(find(np.arange(18).reshape(3, 6), [2, 10, 13], axis=0))
