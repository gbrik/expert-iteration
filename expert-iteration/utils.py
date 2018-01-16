import numpy as np
from typing import List, Tuple, Union, Sequence

def unzip(list_of_tuples: Sequence[Tuple]) -> Tuple[List, ...]:
    unzipped_as_tuples = list(zip(*list_of_tuples))
    return tuple([list(tup) for tup in unzipped_as_tuples])

def sample(arr: np.ndarray) -> Union[int, np.ndarray]:
    with np.errstate(divide='ignore'):
        keys = np.where(arr != 0, np.random.uniform(size=arr.shape) ** (1 / arr), 0)
    return np.argmax(keys, axis=len(arr.shape) - 1)

def softmax(x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if mask != None:
        exped = np.where(mask, np.exp(x), 0)
    else:
        exped = np.exp(x)
    return exped / np.sum(exped, axis=len(exped.shape) - 1).reshape((1,) + exped.shape[:-1]).transpose()
