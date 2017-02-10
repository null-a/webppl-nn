import sys
import json
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

if len(sys.argv) < 2:
    print 'Usage:'
    print '  python show_digit.py <json-file>'
    exit()

def load(fn):
    with open(fn) as f:
        return json.load(f)

img = np.array(load(sys.argv[1])).reshape([28, 28])

plt.imshow(img)
plt.show()
