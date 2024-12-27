import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


def upsample_2d(y, factor=2):
    C, H, W = y.shape
    y = jnp.reshape(y, [C, H, 1, W, 1])
    y = jnp.tile(y, [1, 1, factor, 1, factor])
    return jnp.reshape(y, [C, H * factor, W * factor])


def reparametrize():
    pass


class MalariaVAE(eqx.Module):
    encoder: list
    mean: eqx.nn.Linear
    log_var: eqx.nn.Linear
    decoder: list
    mykey: Array
    hidden_size: int

    def __init__(self, key, hidden_size=2, in_channels=1):
        *keys, self.mykey = jr.split(key, 16)
        self.hidden_size = hidden_size

        self.encoder = [
            eqx.nn.Conv2d(in_channels, 32, kernel_size=5, key=keys[0]),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 32, kernel_size=1, key=keys[1]),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 64, kernel_size=5, stride=2, key=keys[2]),
            eqx.nn.MaxPool2d(kernel_size=4, stride=2),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 64, kernel_size=1, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 128, kernel_size=7, stride=2, key=keys[4]),
            eqx.nn.MaxPool2d(kernel_size=5, stride=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(2048, 512, key=keys[5]),
            jax.nn.relu,
        ]

        self.mean = eqx.nn.Linear(512, hidden_size, key=keys[6])
        self.log_var = eqx.nn.Linear(512, hidden_size, key=keys[7])

        self.decoder = [
            eqx.nn.Linear(hidden_size, 512, key=keys[8]),
            jax.nn.relu,
            eqx.nn.Linear(512, 2048, key=keys[9]),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (128, 4, 4)),
            lambda x: upsample_2d(x, factor=3),
            eqx.nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, key=keys[10]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(64, 64, kernel_size=1, key=keys[11]),
            jax.nn.relu,
            lambda x: upsample_2d(x, factor=2),
            lambda x: jnp.pad(x, ((0, 0), (1, 1), (1, 1))),
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, key=keys[12]),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, 32, kernel_size=1, key=keys[13]),
            jax.nn.relu,
            lambda x: jnp.pad(x, ((0, 0), (0, 1), (0, 1))),
            eqx.nn.ConvTranspose2d(32, in_channels, kernel_size=5, key=keys[14]),
            jax.nn.sigmoid,
        ]

    def __call__(
        self, key, x: Float[Array, "1 128 128"]
    ):  # -> Float[Array, "1 128 128"]:
        mean, log_var = self.encode(x)
        z = self.reparametrize(key, mean, log_var)
        x = self.decode(z)
        return x, z, log_var, mean

    def encode(self, x: Float[Array, "1 128 128"]):
        for layer in self.encoder:
            x = layer(x)

        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var

    def reparametrize(self, key, mean, log_var):
        # generate standard normal random variable
        eps = jr.multivariate_normal(
            key, jnp.zeros(self.hidden_size), jnp.diag(jnp.ones(self.hidden_size))
        )
        # reparametrize to given mean and variance
        z = mean + jnp.exp(log_var / 2) * eps

        return z

    def decode(self, x: Float[Array, "1 2"]):
        for layer in self.decoder:
            x = layer(x)

        return x


if __name__ == "__main__":
    model = MalariaVAE(jr.key(1020))

    from data.malaria import get_dataloaders, load_dataset

    train_data, test_data = load_dataset("autoencoders/data/malaria")
    train_loader, _ = get_dataloaders(train_data, test_data, 64)

    img, label = next(iter(train_loader))

    img = img.numpy()
    label = label.numpy()

    y, h, *_ = jax.vmap(model)(jr.split(jr.key(1234), img.shape[0]), img)

    print(img.shape)
    print(label.shape)
    print(y.shape)
    print(h.shape)
