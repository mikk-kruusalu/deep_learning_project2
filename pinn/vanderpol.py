import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float


class NN(eqx.Module):
    layers: list

    def __init__(self, nhidden, nlayers, key):
        keys = jr.split(key, nlayers)

        hidden_neurons = [1, *((nlayers - 1) * [nhidden]), 1]
        self.layers = [
            eqx.nn.Linear(a, b, key=keys[i])
            for i, (a, b) in enumerate(zip(hidden_neurons[:-1], hidden_neurons[1:]))
        ]

    def __call__(self, t: Float):
        t = jnp.array([t])
        for layer in self.layers[:-1]:
            t = jax.nn.sigmoid(layer(t))

        x = self.layers[-1](t)
        return x.reshape()


@eqx.filter_jit
def loss(model: NN, t: Float[Array, ""], x0: Float, x0d: Float):
    x = jax.vmap(model)(t)
    modeld = jax.grad(model)
    modeldd = jax.grad(modeld)

    u0, u0d = jax.value_and_grad(model)(0.0)
    boundary_loss = (x0 - u0) ** 2 + (x0d - u0d) ** 2 * 1e-1

    xd = jax.vmap(modeld)(t)
    xdd = jax.vmap(modeldd)(t)
    equation_loss = jnp.sum((xdd + 4 * xd + 20**2 * x) ** 2) / t.shape[0] * 1e-4
    return boundary_loss + equation_loss


@eqx.filter_jit
def make_step(model, optim, opt_state, t, x0, x0d):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, t, x0, x0d)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


nn = NN(32, 3, jr.key(1234))
print(nn)

t = jnp.linspace(0, 1, 100)
x0 = 1.0
x0d = 0.0
l = loss(nn, t, x0, x0d)
print(l)

optim = optax.adam(2e-3)
opt_state = optim.init(eqx.filter(nn, eqx.is_array))
for epoch in range(15000):
    nn, opt_state, loss_value = make_step(nn, optim, opt_state, t, x0, x0d)
    print(f"{epoch}: {loss_value}")

plt.plot(t, jax.vmap(nn)(t))
plt.show()
