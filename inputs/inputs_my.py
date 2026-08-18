# Since the functions are in the Sibling's folder, these lines should be added.
#import sys
#sys.path.append("../")
#from functions.functions import *

import logging
logging.getLogger('docplex').setLevel(logging.WARNING) #To avoid unnecessary entries of docplex in the log file
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from json import JSONEncoder
import json
import numpy as np
np.set_printoptions(legacy='1.25') #If not, the prints would be like float64(), int64(), ...
import scipy.integrate as si
import scipy.optimize as so
import docplex.mp
from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader

import numpy as np
from itertools import product
import itertools

import shelve #Like Picke to store the data

# import scipy as sp
# import scipy.stats as spt
import matplotlib.pyplot as plt
from pprint import pprint, pformat
import pandas as pd
from openpyxl import Workbook
from varname import nameof  # get the name of a variable
import random

from tqdm import tqdm  # Progress Bar
import math
import scipy
from scipy import stats
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import MinMaxScaler  # to Normalize

import matplotlib.pyplot as plt
from time import time  # Measure Run Time
from datetime import datetime  # Saving filename as date/time
import os  # To create a folder
import inspect


# region json
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


# endregion


# region Import Inputs
#with open("./inputs/inputs.json", "r") as read_file:
    #inputs = json.load(read_file)
from scipy.stats import truncnorm

def truncated_normal(mean, sd, low, high):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm(a, b, loc=mean, scale=sd)

inputs = {
    "no_basis_fn": 3,
    # Discount
    "gamma": 0.95,

    # State bounds
    "s_min": -10,
    "s_max": 10,

    # Action bounds
    "a_min": 0,
    "a_max": 10,   # not explicitly given; must choose (reasonable upper bound)

    # Demand distribution
    "demand_dist": truncated_normal(mean=5, sd=2, low=0, high=10),

    # Cost parameters
    "cp": 20,   # purchase
    "ch": 2,    # holding
    "cb": 10,   # backlog
    "cd": 10,   # disposal
    "cl": 100,  # lost demand

    # Sampling parameters (for expectations)
    "num_samples": 1000
}

# Assign the value of each Json's key to itself.
for key in inputs:
    if isinstance(inputs[key], list):
        globals()[key] = np.asarray(inputs[key])
    else:
        globals()[key] = inputs[key]
# endregion

# region Import Demands

# with open("./helpers/create_skewed_samples/demands.json", "r") as read_file:
# jsDemands = json.load(read_file)

# Printing in the console with different colors or bold
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def PrintTitle(text):
    print(color.PURPLE + color.BOLD + '─' * 100 + color.END)
    print(color.PURPLE + color.BOLD + text + color.END)
    print(color.PURPLE + color.BOLD + '─' * 100 + color.END)