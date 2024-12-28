from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax
import jax.random as jr


# see also https://docs.kidger.site/equinox/examples/deep_convolutional_gan/
# and https://github.com/rasbt/stat453-deep-learning-ss21/blob/main/L18/04_02_dcgan-celeba.ipynb
class Generator(eqx.Module):
    layers: list[Union[eqx.nn.ConvTranspose2d, eqx.nn.BatchNorm, Callable]]

    def __init__(self, input_shape: int, output_shape: tuple[int, int, int], key):
        keys = jr.split(key, 6)

        height, width, channels = output_shape

        self.layers = [
            eqx.nn.ConvTranspose2d(
                in_channels=input_shape,
                out_channels=width * 16,
                kernel_size=4,
                stride=1,
                padding=0,
                use_bias=False,
                key=keys[0],
            ),
            eqx.nn.BatchNorm(input_size=width * 16, axis_name="batch"),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(
                in_channels=width * 16,
                out_channels=width * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[1],
            ),
            eqx.nn.BatchNorm(input_size=width * 8, axis_name="batch"),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(
                in_channels=width * 8,
                out_channels=width * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[2],
            ),
            eqx.nn.BatchNorm(input_size=width * 4, axis_name="batch"),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(
                in_channels=width * 4,
                out_channels=width * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[3],
            ),
            eqx.nn.BatchNorm(input_size=width * 2, axis_name="batch"),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(
                in_channels=width * 2,
                out_channels=width,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[4],
            ),
            eqx.nn.BatchNorm(input_size=width, axis_name="batch"),
            jax.nn.relu,
            eqx.nn.ConvTranspose2d(
                in_channels=width,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[5],
            ),
            jax.nn.tanh,
        ]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)

        return x, state


class Discriminator(eqx.Module):
    layers: list[Union[eqx.nn.Conv2d, eqx.nn.PReLU, eqx.nn.BatchNorm, Callable]]

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        key,
    ):
        keys = jr.split(key, 6)

        height, width, channels = input_shape

        self.layers = [
            eqx.nn.Conv2d(
                in_channels=channels,
                out_channels=width,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[0],
            ),
            eqx.nn.PReLU(0.2),
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[1],
            ),
            eqx.nn.BatchNorm(width * 2, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            eqx.nn.Conv2d(
                in_channels=width * 2,
                out_channels=width * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[2],
            ),
            eqx.nn.BatchNorm(width * 4, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            eqx.nn.Conv2d(
                in_channels=width * 4,
                out_channels=width * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[3],
            ),
            eqx.nn.BatchNorm(width * 8, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            eqx.nn.Conv2d(
                in_channels=width * 8,
                out_channels=width * 16,
                kernel_size=4,
                stride=2,
                padding=1,
                use_bias=False,
                key=keys[4],
            ),
            eqx.nn.BatchNorm(width * 16, axis_name="batch"),
            eqx.nn.PReLU(0.2),
            eqx.nn.Conv2d(
                in_channels=width * 16,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                use_bias=False,
                key=keys[5],
            ),
        ]

    def __call__(self, x, state):
        for layer in self.layers:
            if isinstance(layer, eqx.nn.BatchNorm):
                x, state = layer(x, state=state)
            else:
                x = layer(x)

        return x, state


if __name__ == "__main__":
    image_size = (128, 128, 1)
    latent_size = 100
    batch_size = 64

    keys = jr.split(jr.key(5467), 3)

    gen = Generator(latent_size, image_size, keys[0])
    gen_state = eqx.nn.State(gen)
    discrim = Discriminator(image_size, keys[1])
    discrim_state = eqx.nn.State(discrim)

    noise = jr.normal(keys[2], (batch_size, latent_size, 1, 1))

    fake_imgs, gen_state = jax.vmap(
        gen, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, gen_state)
    print(fake_imgs.shape)

    pred, discrim_state = jax.vmap(
        discrim, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_imgs, discrim_state)
    print(pred.shape)
