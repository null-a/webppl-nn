import sys
import numpy as np
import json
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

def load(filename):
    with open(filename) as f:
        return json.load(f)

def show_seq(seq):
    # The intention here is that time runs from left to right, and
    # that high notes are at the top, low notes at the bottom.
    plt.imshow(np.flipud(np.vstack(seq).T))
    plt.show()

show_seq(load(sys.argv[1]))
#show_seq(load(sys.argv[1])['train'][0])
