'use strict';

// We rely on the fact that ad is available globally in webppl.
var Tensor = ad.tensor.__Tensor;

// Sum over rows of a matrix.
function _sumreduce0(t) {
  if (t.dims.length !== 2) {
    throw new Error('sumreduce0 is only implemented for matrices.');
  }
  var h = t.dims[0];
  var w = t.dims[1];
  var out = new Tensor([h, 1]);
  for (var i = 0; i < h; i++) {
    for (var j = 0; j < w; j++) {
      out.data[i] += t.data[i * w + j];
    }
  }
  return out;
};

var sumreduce0 = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'sumreduce0',
  forward: function(a) {
    return _sumreduce0(a);
  },
  backward: function(a) {
    var h = a.x.dims[0];
    var w = a.x.dims[1];
    for (var i = 0; i < h; i++) {
      for (var j = 0; j < w; j++) {
        a.dx.data[i * w + j] += this.dx.data[i];
      }
    }
  }
});

module.exports = {
  sumreduce0: sumreduce0,
  _sumreduce0: _sumreduce0
};
