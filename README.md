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
functions, typically from vectors to vectors. (By building
on [adnn](https://github.com/dritchie/adnn).) This package provides a
number of helper functions that capture common patterns in the shape
of these functions. These helpers typically take a name and
input/output dimensions as arguments.

```js
var net = affine('net', {in: 3, out: 5});
var out = net(ones([3, 1])); // dims(out) == [5, 1]
```

Larger networks are built with function composition. The `stack`
helper makes the common pattern of stacking "layers" more readable.

```js
var mlp = stack([
  sigmoid,
  affine('layer2', {in: 5, out: 1}),
  tanh,
  affine('layer1', {in: 5, out: 5})
]);
```

By default, the parameters for such functions are created internally
using the `param` method. An alternative method can be specified using
the `param` argument. For example, the model parameter helpers can be
used here:

```js
var net1 = linear('net1', {in: 20, out: 10, param: modelParam});
var net2 = linear('net2', {in: 20, out: 10, param: modelParamL2(1)});
```

Note that parameters are created when a network constructor (`linear`,
`affine`, etc.) is called. This is a change from earlier versions of
webppl-nn, where parameter creation was delayed until the function
representing the network was applied to an input.

As a consequence, in typical usage, network constructors should now be
called from *within* `Optimize`, rather than from outside of
`Optimize`. See the VAE [example](#examples) to see what this looks
like in practice.

### Model Parameters

WebPPL parameters are primarily used to parameterize *guide* programs.
In the model, the analog of a parameter is a prior guided by a delta
distribution. This choice of guide gives a point estimate of the value
of the random choice in the posterior when performing inference as
optimization.

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

Note that in general, model parameters and parameters created with
`param` are somewhat different in their behavior. For example, these
two fragments of code are equivalent:

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

#### `linear(name, {in, out[, param, init]})`
#### `affine(name, {in, out[, param, initb]})`

These return a parameterized function of a single argument that maps a
vector of length `in` to a vector of length `out`.

#### `bias(name, {out, [param, initb]})`

Returns a parameterized function of a single argument that maps
vectors of length `out` to vectors of length `out`.

#### `rnn(name, {hdim, xdim, [, param, ctor, output]})`
#### `gru(name, {hdim, xdim, [, param, ctor]})`
#### `lstm(name, {hdim, xdim, [, param]})`

These return parameterized function of two arguments that maps a state
vector of length `hdim` and an input vector of length `xdim` to a new
state vector.

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
