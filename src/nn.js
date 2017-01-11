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

function _relu(t) {
  var _t = t;
  var out = new Tensor(_t.dims);
  for (var i = 0; i < _t.length; i++) {
    out.data[i] = _t.data[i] < 0 ? 0 : _t.data[i];
  }
  return out;
}

var relu = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'relu',
  forward: function(t) {
    return _relu(t);
  },
  backward: function(x) {
    for (var i = 0; i < x.x.length; i++) {
      x.dx.data[i] += x.x.data[i] < 0 ? 0 : this.dx.data[i];
    }
  }
});

var leakyness = 100;

function _lrelu(t) {
  var _t = t;
  var out = new Tensor(_t.dims);
  for (var i = 0; i < _t.length; i++) {
    out.data[i] = _t.data[i] < 0 ? _t.data[i] / leakyness : _t.data[i];
  }
  return out;
}

var lrelu = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'lrelu',
  forward: function(t) {
    return _relu(t);
  },
  backward: function(x) {
    for (var i = 0; i < x.x.length; i++) {
      x.dx.data[i] += x.x.data[i] < 0 ? this.dx.data[i] / leakyness : this.dx.data[i];
    }
  }
});

function oneHot(index, length) {
  if (length <= 0) {
    throw new Error('length should be > 0.');
  }
  if (index < 0 || index >= length) {
    throw new Error('index out of bounds');
  }
  var out = new Tensor([length, 1]);
  out.data[index] = 1;
  return out;
}

module.exports = {
  sumreduce0: sumreduce0,
  _sumreduce0: _sumreduce0,
  relu: relu,
  lrelu: lrelu,
  oneHot: oneHot
};
