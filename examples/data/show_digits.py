import sys
import json
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

if len(sys.argv) < 2:
    print 'Usage:'
    print '  python show_digits.py <json-file>'
    exit()

def load(fn):
    with open(fn) as f:
        return json.load(f)

def tile(X, grid_shape=None, spacing=1, channel_count=1):
    tile_length = X.shape[1] / channel_count
    tile_size = int(np.sqrt(tile_length))

    # Layout all the cases in a square grid if grid_shape is not
    # specified.
    if grid_shape is None:
        sqrt_num_cases = np.sqrt(X.shape[0])
        if sqrt_num_cases.is_integer() and sqrt_num_cases < 25:
            grid_shape = int(sqrt_num_cases)

    assert grid_shape is not None

    grid_shape = np.array(grid_shape)

    if grid_shape.size == 1:
        # Assume square grid is grid_shape wasn't a list/tuple.
        grid_shape = grid_shape.repeat(2)

    assert grid_shape.shape == (2,)

    out_shape = (tile_size + spacing) * grid_shape - spacing
    if channel_count > 1:
        out_shape = tuple(out_shape) + (channel_count,)
    out_image = np.ones(out_shape)

    if channel_count == 1:
        for y in xrange(grid_shape[0]):
            for x in xrange(grid_shape[1]):
                t = np.reshape(X[y*grid_shape[1]+x,:], (tile_size, tile_size))
                out_image[
                    y*(tile_size+spacing):(y*(tile_size+spacing)+tile_size),
                    x*(tile_size+spacing):(x*(tile_size+spacing)+tile_size)
                    ] = t
    else:
        for i in range(channel_count):
            out_image[:,:,i] = tile(X[:,i*tile_length:(i+1)*tile_length],
                                    grid_shape, spacing)

    return out_image

plt.imshow(tile(np.array(load(sys.argv[1]))))
plt.show()
