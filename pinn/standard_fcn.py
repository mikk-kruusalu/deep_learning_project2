import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Float


class FCN(eqx.Module):
    """Standard fully connected neural network, which has `nlayers` and
    each layer has `nhidden` neurons. The input is a scalar and the output
    is scalar as well.
    """

    layers: list

    def __init__(self, nhidden, nlayers, key):
        keys = jr.split(key, nlayers)

        hidden_neurons = [1, *((nlayers - 1) * [nhidden]), 1]
        self.layers = [
            eqx.nn.Linear(a, b, key=keys[i])
            for i, (a, b) in enumerate(zip(hidden_neurons[:-1], hidden_neurons[1:]))
        ]

    def __call__(self, t: Float) -> Float:
        t = t.reshape(1)
        for layer in self.layers[:-1]:
            t = jax.nn.tanh(layer(t))

        x = self.layers[-1](t)
        return x.reshape()
