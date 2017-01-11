import cPickle
import numpy as np
import json

# http://www-etud.iro.umontreal.ca/~boulanni/icml2012
files = ['JSB-Chorales',
         'MuseData',
         'Nottingham',
         'Piano-midi.de']

def load(filename):
    with open(filename) as f:
        return cPickle.load(f)

def tup2list(tup):
    # Map a tuple of notes (active at a time step), to a vector.
    out = np.zeros([88], dtype=int)
    if len(tup) > 0:
        assert min(tup) >= 21, 'Unexpected value in tuple.'
        assert max(tup) <= 108, 'Unexpected value in tuple.'
        out[np.array([n - 21 for n in tup])] = 1
    return out.tolist()

#print tup2list((21, 108))
#print tup2list(tuple())

# Format:
# {'test': [[(n1, n2, ...), (...), ...], [], ...], 'train': [], 'valid': []}

def convert(data):
    return dict((name,[[tup2list(step) for step in seq] for seq in dataset]) for (name,dataset) in data.iteritems())

for f in files:
    data = convert(load(f + '.pickle'))
    with open(f + '.json', 'w') as f:
        json.dump(data, f)
