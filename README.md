# webppl-nn

## Installation

To globally install `webppl-nn`, run:

    mkdir -p ~/.webppl
    npm install --prefix ~/.webppl https://github.com/null-a/webppl-nn

## Usage

Once installed, you can make the package available to `program.wppl`
by running:

    webppl program.wppl --require webppl-nn

## Status

This package is very experimental. Expect frequent breaking changes.

## Compatibility

This package currently requires the development version of WebPPL.
(i.e. The tip of the `dev` branch.)

## Introduction

### Neural Nets

In WebPPL we can represent "neural" networks as parameterized
functions, typically from vectors to vectors. (By building on
[adnn](https://github.com/dritchie/adnn).) This package provides a
number of helper functions that capture common patterns in the shape
of these functions. These helpers typically take a name and an output
dimension as arguments.

```js
var net = affine('net', {out: 5});
var out = net(ones([3, 1])); // dims(out) == [5, 1]
```

Larger networks are built with function composition. The `stack`
helper makes the common pattern of stacking "layers" more readable.

```js
var mlp = stack([
  sigmoid,
  affine('layer2', {out: 1}),
  tanh,
  affine('layer1', {out: 5})
]);
```

Such functions need to be parameterized by either guide or model
parameters depending on where they are used. By default, the networks
created with these helpers are parameterized directly by guide
parameters. When the network is intended for use in the model, one of
the model parameter helpers described above can be passed to the
network constructor.

```js
var guideNet = linear('net1', {out: 10});
var modelNet = linear('net1', {out: 10, param: modelParamL2(1)});
```

### Model Parameters

WebPPL "parameters" are primarily used to parameterize *guide*
programs. In the model, the analog of a parameter is a prior guided by
a delta distribution. This choice of guide gives a point estimate of
the value of the random choice in the posterior when performing
inference as optimization.

WebPPL includes a helper `modelParam` which creates model parameters
using an improper uniform distribution as the prior. Since it is not
possible to sample from this improper distribution `modelParam` can
only be used with optimization based algorithms.

This package provides an additional helper `modelParamL2` which can be
used to create model parameters that have a Gaussian prior. When
performing inference as optimization this prior acts as a regularizer.
Since `modelParamL2` creates a Gaussian random choice, it can be used
with all sampling based inference algorithms.

To allow the width of the prior to be specified, `modelParamL2` takes
a single argument specifying the standard deviation of the Gaussian.
This returns a function that takes an object in the same format as
`param` and `modelParam`.

```js
var w = modelParamL2(1)({name: 'w', dims: [2, 2]});
```

Note that in general, there is a subtle difference in the behavior of
model parameters and parameters created with `param`.

For example, with `param`, these two fragments of code are equivalent:

```js
// 1.
var p = param({name: 'p'});
f(p);
g(p);

// 2.
f(param({name: 'p'}))
g(param({name: 'p'}))
```

However, if `param({name: 'p'})` is replaced with
`modelParamL2(1)({name: 'p'})` for example, then they are *not*
equivalent. The reason is that each call to `modelParamL2()` adds a
random choice to the model. In the common setting of optimizing the
ELBO for example, each such random choice has the effect of extending
the optimization objective with a weight decay term for its parameter.
i.e. Additional calls to `modelParamL2()` (for a particular parameter)
incur additional weight decay penalties.

## Examples

* [Variational Auto-encoder](https://github.com/null-a/webppl-nn/blob/master/examples/vae.wppl)

## Reference

### Networks

#### `linear(name, {out[, param, init]})`
#### `affine(name, {out[, param, initb]})`

These return a parameterized function of a single argument. This
function maps a vector to a vector of length `nout`.

#### `bias(name[, {param, initb}])`

Returns a parameterized function of a single argument. This function
maps vectors of length `n` to vectors of length `n`.

#### `rnn(name, {dim[, param, ctor, output]})`
#### `gru(name, {dim[, param, ctor]})`
#### `lstm(name, {dim[, param]})`

These return parameterized function of two arguments. This function
maps a state vector and an input vector to a new state vector.

### Non-linearities

#### `sigmoid(x)`
#### `tanh(x)`
#### `relu(x)`
#### `lrelu(x)`

Leaky rectified linear unit.

#### `softplus(x)`

#### `softmax(x)`
#### `squishToProbSimplex(x)`

Maps vectors of length `n` to probability vectors of length `n + 1`.

In contrast to the `softmax` function, a network with
`squishToProbSimplex` at the output and no regularization is not over
parameterized. However, with regularization, a network with `softmax`
at the output will not be over parameterized either.

<!--

Using squishToProbSimplex to with a prior on the parameters centered
at zero seems a bit fishy. For example, these two output the same
vector only with the elements permuted:

squishToProbSimplex(vec([-1,-1,-1]))
squishToProbSimplex(vec([1,0,0]))

... yet under a Gaussian prior they aren't equally likely. Something
similar applies when using regularization.

-->

### Model Parameters

#### `modelParamL2(sd)`

Returns a function that creates model parameters with a `Gaussian({mu:
0, sigma: sd})` prior. The returned function has the same interface as
`param` and `modelParam`.

### Utilities

#### `stack(fns)`

Returns the composition of the array of functions `fns`. The functions
in `fns` are applied in right to left order.

#### `idMatrix(n)`

Returns the `n` by `n` identity matrix.

#### `oneHot(index, length)`

Returns a vector with length `length` in which all entries are zero
except for the entry at `index` which is one.

#### `concat(arr)`

Returns the vector obtained by concatenating the elements of `arr`.
(`arr` is assumed to be an array of vectors.)

```js
concat([ones([2, 1]), zeros([2, 1])]); // => Vector([1, 1, 0, 0])
```

## License

MIT
