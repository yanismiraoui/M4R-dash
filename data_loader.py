"""
Code for loading data
"""

from typing import *

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import Levenshtein

import featurization as ft
import utils

# Names of datasets
DATASET_NAMES = {"LCMV", "VDJdb", "PIRD", "TCRdb"}


def chunkify(x: Sequence[Any], chunk_size: int = 128):
    """
    Split list into chunks of given size
    >>> chunkify([1, 2, 3, 4, 5, 6, 7], 3)
    [[1, 2, 3], [4, 5, 6], [7]]
    >>> chunkify([(1, 10), (2, 20), (3, 30), (4, 40)], 2)
    [[(1, 10), (2, 20)], [(3, 30), (4, 40)]]
    """
    retval = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    return retval
