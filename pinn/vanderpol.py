import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jaxtyping import Array, Float
from standard_fcn import FCN


@eqx.filter_jit
def loss(model: FCN, t: Float[Array, ""], x0: Float, x0d: Float, mu: Float):
    x = jax.vmap(model)(t)
    modeld = jax.grad(model)
    modeldd = jax.grad(modeld)

    u0, u0d = jax.value_and_grad(model)(0.0)
    boundary_loss = (x0 - u0) ** 2 + (x0d - u0d) ** 2 * 1e-1

    xd = jax.vmap(modeld)(t)
    xdd = jax.vmap(modeldd)(t)
    equation_loss = jnp.mean((xdd - mu * (1 - x**2) * xd + x) ** 2) * 1e-1
    return boundary_loss + equation_loss


@eqx.filter_jit
def make_step(model, optim, opt_state, t, x0, x0d, mu):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, t, x0, x0d, mu)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def solve_pinn(nn, t, x0, x0d, mu):
    optim = optax.adamw(1e-2, nesterov=True, weight_decay=0.001)
    opt_state = optim.init(eqx.filter(nn, eqx.is_array))

    for epoch in range(15001):
        nn, opt_state, loss_value = make_step(nn, optim, opt_state, t, x0, x0d, mu)
        if epoch % 500 == 0:
            print(f"{epoch}: {loss_value}")

    return nn


# solve with Diffrax
def solve_diffrax(t, x0, x0d, mu):
    def dSdt(t, S, p):
        x, y = S
        return y, mu * (1 - x**2) * y - x

    sol = diffeqsolve(
        ODETerm(dSdt),
        Tsit5(),
        t[0],
        t[-1],
        0.01,
        (x0, x0d),
        args=mu,
        saveat=SaveAt(dense=True),
    )
    return sol


@eqx.filter_jit
def fit_loss(pytree: tuple[FCN, Array], t: Float[Array, ""], data: Float[Array, "2 "]):
    model, mu = pytree

    x = jax.vmap(model)(t)
    modeld = jax.grad(model)
    modeldd = jax.grad(modeld)

    lsq_loss = jnp.mean((jax.vmap(model)(data[0]) - data[1]) ** 2)

    xd = jax.vmap(modeld)(t)
    xdd = jax.vmap(modeldd)(t)
    equation_loss = jnp.mean((xdd - mu * (1 - x**2) * xd + x) ** 2)
    return lsq_loss + equation_loss


@eqx.filter_jit
def make_step_fit(pytree, optim, opt_state, t, data):
    loss_value, grads = eqx.filter_value_and_grad(fit_loss)(pytree, t, data)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(pytree, eqx.is_array)
    )
    pytree = eqx.apply_updates(pytree, updates)
    return pytree, opt_state, loss_value


def fit_pinn(nn, t, data, mu):
    optim = optax.adam(1e-3)
    pytree = (nn, jnp.array(mu))
    opt_state = optim.init(eqx.filter(pytree, eqx.is_array))

    for epoch in range(15001):
        pytree, opt_state, loss_value = make_step_fit(pytree, optim, opt_state, t, data)
        if epoch % 500 == 0:
            print(f"{epoch}: {loss_value} \t mu: {pytree[1]}")

    return pytree


if __name__ == "__main__":
    # set initial conditions
    t = jnp.linspace(0, 8, 100)
    x0 = 2.0
    x0d = 0.0
    mu = 1.2

    # solve the equation
    nn = solve_pinn(FCN(32, 3, jr.key(1234)), t, x0, x0d, mu)
    sol = solve_diffrax(t, x0, x0d, mu)

    # generate random data
    t_rand = 5.0 * jr.uniform(jr.key(2645), shape=(20,))
    y_rand = jax.vmap(sol.evaluate)(t_rand)[0] + 0.1 * jr.normal(
        jr.key(6724), shape=(20,)
    )
    data = jnp.stack([t_rand, y_rand])
    nn_fit, mu_fit = fit_pinn(FCN(32, 3, jr.key(5432)), t, data, 0.0)

    plt.plot(t, jax.vmap(nn)(t), label="PINN")
    plt.plot(t, jax.vmap(sol.evaluate)(t)[0], label="Diffrax")  # type: ignore
    plt.scatter(t_rand, y_rand, label="Noisy data", c="tab:green")
    plt.plot(t, jax.vmap(nn_fit)(t), label="Fitted PINN")
    plt.legend()
    plt.show()
