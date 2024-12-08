import argparse
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
import yaml
from autoencoder import MalariaAutoencoder
from data.malaria import get_dataloaders, load_dataset
from jaxtyping import Array, Float
from torch.utils.data import DataLoader


@eqx.filter_jit
def loss(
    model: MalariaAutoencoder,
    img: Float[Array, "batch 1 128 128"],
) -> Float[Array, ""]:
    pred, h = jax.vmap(model)(img)

    reconstruction_loss = jnp.sum((pred - img) ** 2)
    hidden_l1 = jnp.sum(jnp.abs(h))

    return reconstruction_loss + hidden_l1


def plot_latent_space(model, loader):
    fig, ax = plt.subplots(constrained_layout=True)

    labels = []
    hiddens = []
    for img, label in loader:
        _, h = jax.vmap(model)(img.numpy())
        hiddens.append(h)
        labels.append(label.numpy())

    labels = jnp.concat(labels)
    hiddens = jnp.concat(hiddens)

    ax.scatter(hiddens[:, 0], hiddens[:, 1], c=labels)
    return fig


def train(
    model: MalariaAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optim: optax.GradientTransformation,
    nepochs: int,
) -> MalariaAutoencoder:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, imgs: Float[Array, "batch 1 128 128"]):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, imgs)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(nepochs):
        train_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.numpy()
            labels = labels.numpy()

            model, opt_state, loss_value = make_step(model, opt_state, imgs)
            train_loss += loss_value.item() / len(train_loader)

        test_loss = 0
        for imgs, labels in test_loader:
            imgs = imgs.numpy()
            labels = labels.numpy()
            test_loss += loss(model, imgs).item() / len(test_loader)

        log: dict[str, Any] = {"train_loss": train_loss, "test_loss": test_loss}
        if epoch % 10 == 0 or epoch == nepochs:
            print(f"Train loss: {train_loss:.4f}  \t test loss: {test_loss:.4f}")

            log.update(
                {
                    "latent_train": wandb.Image(plot_latent_space(model, train_loader)),
                    "latent_test": wandb.Image(plot_latent_space(model, test_loader)),
                }
            )
        wandb.log(log)

    wandb.finish()

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    hyperparams = config["hyperparams"]

    wandb.init(
        project=config["logger"]["project"],
        name=config["exp_name"],
        config=config["hyperparams"],
    )

    train_data, test_data = load_dataset("autoencoders/data/malaria")
    train_loader, test_loader = get_dataloaders(
        train_data, test_data, hyperparams["batch_size"]
    )

    optim = optax.adamw(hyperparams["learning_rate"])
    model = MalariaAutoencoder(key=jax.random.key(1234), **hyperparams["model"])

    train(model, train_loader, test_loader, optim, hyperparams["nepochs"])
