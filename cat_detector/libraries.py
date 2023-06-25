import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *
import time

%matplotlib inline
%load_ext autoreload
%autoreload 2

""" 
    numpy is the fundamental package for scientific computing with Python.
    h5py is a common package to interact with a dataset that is stored on an H5 file.
    matplotlib is a famous library to plot graphs in Python.
    PIL and scipy are used here to test your model with your own picture at the end.
"""