import argparse
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import wandb
import yaml
from data.satellite_images import get_dataloaders, load_dataset
from gans import Discriminator, Generator
from jaxtyping import Array, Float
from torch.utils.data import DataLoader


@eqx.filter_value_and_grad(has_aux=True)
def loss_discriminator(
    discriminator,
    generator,
    real_imgs: Float[Array, "batch 3 64 64"],
    gen_state,
    discriminator_state,
    latent_size,
    key: Float[Array, " "],
):
    batch_size = real_imgs.shape[0]
    key, subkey = jr.split(key)
    noise = jr.normal(subkey, (batch_size, latent_size, 1, 1))

    fake_imgs, gen_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, gen_state)

    fake_labels = jnp.zeros(batch_size)
    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_imgs, discriminator_state)
    loss1 = optax.sigmoid_binary_cross_entropy(pred_y, fake_labels).mean()

    real_labels = jnp.ones(batch_size)
    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(real_imgs, discriminator_state)
    loss2 = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    loss = (loss1 + loss2) / 2

    return loss, (discriminator_state, gen_state, key)


@eqx.filter_value_and_grad(has_aux=True)
def loss_generator(
    generator,
    discriminator,
    discriminator_state,
    gen_state,
    batch_size,
    latent_size,
    key,
):
    key, subkey = jr.split(key)
    noise = jr.normal(subkey, (batch_size, latent_size, 1, 1))

    fake_imgs, gen_state = jax.vmap(
        generator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(noise, gen_state)

    real_labels = jnp.ones(batch_size)
    pred_y, discriminator_state = jax.vmap(
        discriminator, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )(fake_imgs, discriminator_state)
    loss = optax.sigmoid_binary_cross_entropy(pred_y, real_labels).mean()

    return loss, (discriminator_state, gen_state, key)


@eqx.filter_jit
def step_discriminator(
    discriminator: Discriminator,
    generator: Generator,
    real_imgs: jnp.ndarray,
    discriminator_optimizer: optax.GradientTransformation,
    discriminator_opt_state: optax.OptState,
    gen_state: eqx.nn.State,
    discriminator_state: eqx.nn.State,
    latent_size,
    key,
):
    (
        (
            loss,
            (discriminator_state, generator_state, key),
        ),
        grads,
    ) = loss_discriminator(
        discriminator,
        generator,
        real_imgs,
        gen_state,
        discriminator_state,
        latent_size,
        key,
    )

    updates, opt_state = discriminator_optimizer.update(grads, discriminator_opt_state)
    discriminator = eqx.apply_updates(discriminator, updates)

    return loss, discriminator, opt_state, generator_state, discriminator_state


@eqx.filter_jit
def step_generator(
    generator: Generator,
    discriminator: Discriminator,
    gen_optimizer: optax.GradientTransformation,
    gen_opt_state: optax.OptState,
    discriminator_state: eqx.nn.State,
    gen_state: eqx.nn.State,
    batch_size,
    latent_size,
    key,
):
    (
        (
            loss,
            (discriminator_state, gen_state, key),
        ),
        grads,
    ) = loss_generator(
        generator,
        discriminator,
        discriminator_state,
        gen_state,
        batch_size,
        latent_size,
        key,
    )

    updates, opt_state = gen_optimizer.update(grads, gen_opt_state)
    generator = eqx.apply_updates(generator, updates)

    return loss, generator, opt_state, discriminator_state, gen_state


def train(
    generator,
    discriminator,
    data_loader: DataLoader,
    gen_optimizer: optax.GradientTransformation,
    discriminator_optimizer: optax.GradientTransformation,
    latent_size: int,
    nepochs: int,
):
    gen_state = eqx.nn.State(generator)
    discriminator_state = eqx.nn.State(discriminator)

    gen_opt_state = gen_optimizer.init(eqx.filter(generator, eqx.is_array))
    discriminator_opt_state = discriminator_optimizer.init(
        eqx.filter(discriminator, eqx.is_array)
    )

    key = jr.key(1026)

    for epoch in range(nepochs):
        log_summary = epoch % 2 == 0 or epoch == nepochs - 1
        gen_loss = 0
        discriminator_loss = 0

        for real_imgs, labels in data_loader:
            batch_size = real_imgs.shape[0]
            keys = jr.split(key, 2)
            key = keys[0]
            real_imgs = real_imgs.numpy()
            labels = labels.numpy()

            (
                loss,
                discriminator,
                discriminator_opt_state,
                generator_state,
                discriminator_state,
            ) = step_discriminator(
                discriminator,
                generator,
                real_imgs,
                discriminator_optimizer,
                discriminator_opt_state,
                gen_state,
                discriminator_state,
                latent_size,
                keys[1],
            )
            discriminator_loss += loss / len(data_loader)

            (
                loss,
                generator,
                gen_opt_state,
                discriminator_state,
                gen_state,
            ) = step_generator(
                generator,
                discriminator,
                gen_optimizer,
                gen_opt_state,
                discriminator_state,
                gen_state,
                batch_size,
                latent_size,
                keys[2],
            )
            gen_loss += loss / len(data_loader)

        log: dict[str, Any] = {
            "train_loss": gen_loss,
            "test_loss": discriminator_loss,
        }
        if log_summary:
            print(
                f"{epoch}: generator loss: {gen_loss:.4f}\t "
                f"discriminator loss: {discriminator_loss:.4f}"
            )

            keys = jr.split(key, 3)
            key = keys[0]
            noise = jr.normal(keys[1], (12, latent_size, 1, 1))
            rand_fake_imgs, gen_state = generator(noise, gen_state)

            fig, axes = plt.subplots(3, 4, constrained_layout=True)
            for ax, fake_img in zip(axes.ravel(), rand_fake_imgs):
                ax.imshow(fake_img)

            log.update({"generator_examples": wandb.Image(fig)})
            plt.close("all")
        wandb.log(log)

    wandb.finish()

    return generator, discriminator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        assert config["model"] == "GAN"
    hyperparams = config["hyperparams"]

    wandb.init(
        project=config["logger"]["project"],
        name=config["exp_name"],
        config=config["hyperparams"],
    )

    keys = jr.split(jr.key(5480), 3)

    train_data, test_data = load_dataset("autoencoders/data/malaria")
    data_loader, _ = get_dataloaders(train_data, test_data, hyperparams["batch_size"])

    gen_optim = optax.adam(hyperparams["learning_rate"])
    discriminator_optim = optax.adam(hyperparams["learning_rate"])

    image_size = (64, 64, 3)
    generator = Generator(hyperparams["model"]["latent_size"], image_size, keys[0])
    discriminator = Discriminator(image_size, keys[1])

    train(
        generator,
        discriminator,
        data_loader,
        gen_optim,
        discriminator_optim,
        hyperparams["model"]["latent_size"],
        hyperparams["nepochs"],
    )
