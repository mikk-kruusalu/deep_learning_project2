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
def loss(
    model: FCN, t: Float[Array, ""], x0: Float, x0d: Float, mu: Float, omega: Float
):
    """The loss function describes a harmonic oscillator equation
    $u'' + \\mu u' + \\omega^2 u = 0$
    subject to initial conditions
    $u(0) = x_0 \\quad u'(0) = u'_0$
    """
    x = jax.vmap(model)(t)
    modeld = jax.grad(model)
    modeldd = jax.grad(modeld)

    u0, u0d = jax.value_and_grad(model)(0.0)
    boundary_loss = (x0 - u0) ** 2 + (x0d - u0d) ** 2 * 1e-1

    xd = jax.vmap(modeld)(t)
    xdd = jax.vmap(modeldd)(t)
    equation = xdd + mu * xd + omega**2 * x
    equation_loss = jnp.sum(equation**2) / t.shape[0] * 1e-4
    return boundary_loss + equation_loss


@eqx.filter_jit
def make_step(model, optim, opt_state, t, x0, x0d, mu, omega):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, t, x0, x0d, mu, omega)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def solve_pinn(nn, t, x0, x0d, mu, omega):
    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(nn, eqx.is_array))

    for epoch in range(15001):
        nn, opt_state, loss_value = make_step(
            nn, optim, opt_state, t, x0, x0d, mu, omega
        )
        if epoch % 500 == 0:
            print(f"{epoch}: {loss_value}")

    return nn


def solve_diffrax(t, x0, x0d, mu, omega):
    def dSdt(t, S, p):
        x, y = S
        return y, -mu * y - omega**2 * x

    sol = diffeqsolve(
        ODETerm(dSdt),
        Tsit5(),
        t[0],
        t[-1],
        0.01,
        (x0, x0d),
        args=(mu, omega),
        saveat=SaveAt(ts=t),
    )
    return sol


def exact(t, mu, omega):
    return jnp.exp(-mu * t / 2) * (jnp.cos(jnp.sqrt(omega**2 - mu**2 / 2) * t))


@eqx.filter_jit
def fit_loss(
    pytree: tuple[FCN, Array], t: Float[Array, ""], data: Float[Array, ""], omega: Float
):
    model, mu = pytree

    x = jax.vmap(model)(t)
    modeld = jax.grad(model)
    modeldd = jax.grad(modeld)

    lsq_loss = jnp.mean((jax.vmap(model)(data[0]) - data[1]) ** 2) * 1e4

    xd = jax.vmap(modeld)(t)
    xdd = jax.vmap(modeldd)(t)
    equation = xdd + mu * xd + omega**2 * x
    equation_loss = jnp.mean(equation**2)
    return lsq_loss + equation_loss


@eqx.filter_jit
def make_step_fit(pytree, optim, opt_state, t, data, omega):
    loss_value, grads = eqx.filter_value_and_grad(fit_loss)(pytree, t, data, omega)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(pytree, eqx.is_array)
    )
    pytree = eqx.apply_updates(pytree, updates)
    return pytree, opt_state, loss_value


def fit_pinn(nn, t, data, mu, omega):
    optim = optax.adam(1e-3)
    pytree = (nn, jnp.array(mu))
    opt_state = optim.init(eqx.filter(pytree, eqx.is_array))

    for epoch in range(15001):
        pytree, opt_state, loss_value = make_step_fit(
            pytree, optim, opt_state, t, data, omega
        )
        if epoch % 500 == 0:
            print(f"{epoch}: {loss_value} \t mu: {pytree[1]}")

    return pytree


if __name__ == "__main__":
    # set initial conditions
    t = jnp.linspace(0, 1, 500)
    x0 = 1.0
    x0d = 0.0
    mu = 4.0
    omega = 20.0

    sol = solve_diffrax(t, x0, x0d, mu, omega)
    nn = solve_pinn(FCN(32, 3, jr.key(1020)), t, x0, x0d, mu, omega)

    # generate noisy data
    t_obs = 0.5 * jr.uniform(jr.key(10), shape=(40,))
    x_obs = exact(t_obs, mu, omega) + 0.04 * jr.normal(jr.key(10), shape=t_obs.shape)
    data = jnp.array([t_obs, x_obs])

    nn_fit, mu_fit = fit_pinn(FCN(32, 3, jr.key(1020)), t, data, 0.0, omega)

    # plt.plot(t, jax.vmap(nn)(t), label="PINN")
    plt.plot(t, sol.ys[0], label="Diffrax")  # type: ignore
    plt.plot(t, exact(t, mu, omega), label="Exact")
    plt.scatter(data[0], data[1], label="data")
    plt.plot(t, jax.vmap(nn_fit)(t), label="fitted PINN")
    plt.legend()
    plt.show()
