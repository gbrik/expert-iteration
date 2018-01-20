import numpy as np
from hashlib import sha1
from typing import List, Tuple, Union, Sequence

def sgn(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def unzip(list_of_tuples: Sequence[Tuple]) -> Tuple[List, ...]:
    unzipped_as_tuples = list(zip(*list_of_tuples))
    return tuple([list(tup) for tup in unzipped_as_tuples])

def sample(arr: np.ndarray) -> Union[int, np.ndarray]:
    with np.errstate(divide='ignore'):
        keys = np.where(arr != 0, np.random.uniform(size=arr.shape) ** (1 / arr), 0)
    return np.argmax(keys, axis=len(arr.shape) - 1)

def softmax(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        exped = np.where(mask, np.exp(x), 0)
    else:
        exped = np.exp(x)
    return exped / np.sum(exped, axis=len(exped.shape) - 1).reshape((1,) + exped.shape[:-1]).transpose()

def rescale(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x, axis=len(x.shape) - 1).reshape((1,) + x.shape[:-1]).transpose()

def hash_arr(x: np.ndarray) -> int:
    return int(sha1(x.view(np.uint8)).hexdigest(), 16)

def histogram(l: List):
    d = dict()
    for x in l:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1
    return d
