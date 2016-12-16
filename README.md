# webppl-nn

## Installation

To globally install `webppl-nn`, run:

    mkdir -p ~/.webppl
    npm install --prefix ~/.webppl https://github.com/null-a/webppl-nn

## Usage

Once installed, you can make the package available to `program.wppl`
by running:

    webppl program.wppl --require webppl-nn

## Docs

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

### Neural Nets

In WebPPL we can represent "neural" networks as parameterized
functions, typically from vectors to vectors. (By building on
[adnn](https://github.com/dritchie/adnn).) This package provides a
number of helper functions that capture common patterns in the shape
of these functions. These helpers typically take an output dimension
and name as arguments.

```js
var net = affine(5, 'net');
var out = net(ones([3, 1])); // dims(out) == [5, 1]
```

Larger networks are built with function composition. The `stack`
helper makes the common pattern of stacking "layers" more readable.

```js
var mlp = stack([
  affine(5, 'layer1'),
  tanh,
  affine(1, 'layer2'),
  sigmoid
]);
```

Such functions need be parameterized by either guide or model
parameters depending on where they are used. By default, the networks
created with these helpers are parameterized directly by guide
parameters. When the network is intended for use in the model, one of
the model parameter helpers described above can be passed to the
network constructor.

```js
var guideNet = linear(10, 'net1');
var modelNet = linear(10, 'net1', modelParamL2(1));
```

The following helpers are available:

`stack`

`sigmoid`
`tanh`

`linear(nout, name{, paramModel})`
`affine(nout, name{, paramModel})`
`bias(name{, paramModel, initialBias})`
`rnn(nout, name{, paramModel, netConstructor, nonLinearity})`
`gru(nout, name{, paramModel, netConstructor})`
`lstm(nout, name{, paramModel})`

## License

MIT
