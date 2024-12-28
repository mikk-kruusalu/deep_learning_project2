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


class MalariaAutoencoder(eqx.Module):
    encoder: list
    decoder: list

    def __init__(self, key, hidden_size=2, in_channels=1):
        (
            key1,
            key2,
            key3,
            key4,
            key5,
            key6,
            key7,
            dkey1,
            dkey2,
            dkey3,
            dkey4,
            dkey5,
            dkey6,
            dkey7,
        ) = jr.split(key, 14)

        self.encoder = [
            eqx.nn.Conv2d(in_channels, 32, kernel_size=5, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 32, kernel_size=1, key=key2),
            jax.nn.relu,
            eqx.nn.Conv2d(32, 64, kernel_size=5, stride=2, key=key3),
            eqx.nn.MaxPool2d(kernel_size=4, stride=2),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 64, kernel_size=1, key=key4),
            jax.nn.relu,
            eqx.nn.Conv2d(64, 128, kernel_size=7, stride=2, key=key5),
            eqx.nn.MaxPool2d(kernel_size=5, stride=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(2048, 512, key=key6),
            jax.nn.relu,
            eqx.nn.Linear(512, hidden_size, key=key7),
        ]

        self.decoder = [
            eqx.nn.Linear(hidden_size, 512, key=dkey7),
            jax.nn.relu,
            eqx.nn.Linear(512, 2048, key=dkey6),
            jax.nn.relu,
            lambda x: jnp.reshape(x, (128, 4, 4)),
            lambda x: upsample_2d(x, factor=3),
            eqx.nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, key=dkey5),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(64, 64, kernel_size=1, key=dkey4),
            jax.nn.relu,
            lambda x: upsample_2d(x, factor=2),
            lambda x: jnp.pad(x, ((0, 0), (1, 1), (1, 1))),
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, key=dkey3),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(32, 32, kernel_size=1, key=dkey2),
            jax.nn.relu,
            lambda x: jnp.pad(x, ((0, 0), (0, 1), (0, 1))),
            eqx.nn.ConvTranspose2d(32, in_channels, kernel_size=5, key=dkey1),
            jax.nn.sigmoid,
        ]

    def __call__(
        self, key, x: Float[Array, "1 128 128"]
    ):  # -> Float[Array, "1 128 128"]:
        for layer in self.encoder:
            x = layer(x)

        hidden = jnp.copy(x)

        for layer in self.decoder:
            x = layer(x)
        return x, hidden


if __name__ == "__main__":
    model = MalariaAutoencoder(jr.key(1020))

    from data.malaria import get_dataloaders, load_dataset

    train_data, test_data = load_dataset("autoencoders/data/malaria")
    train_loader, _ = get_dataloaders(train_data, test_data, 64)

    img, label = next(iter(train_loader))

    img = img.numpy()
    label = label.numpy()

    y, h = jax.vmap(model, in_axes=(None, 0))(jr.key(102), img)

    print(img.shape)
    print(label.shape)
    print(y.shape)
    print(h.shape)
