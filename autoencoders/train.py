import argparse
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
import yaml
from autoencoder import MalariaAutoencoder
from data.malaria import get_dataloaders, load_dataset
from jaxtyping import Array, Float
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from vae import MalariaVAE


@eqx.filter_jit
def loss_ae(
    model: MalariaAutoencoder,
    key: Float[Array, " "],
    img: Float[Array, "batch 1 128 128"],
) -> Float[Array, ""]:
    pred, h = jax.vmap(model)(key, img)

    reconstruction_loss = jnp.sum((pred - img) ** 2)
    hidden_l1 = jnp.sum(jnp.abs(h))

    return reconstruction_loss + hidden_l1


@eqx.filter_jit
def loss_vae(
    model: MalariaVAE,
    key: Float[Array, "batch 1"],
    img: Float[Array, "batch 1 128 128"],
) -> Float[Array, ""]:
    pred, h, log_var, mu = jax.vmap(model)(key, img)

    reconstruction_loss = jnp.sum((pred - img) ** 2)
    kl_div = -0.5 * jnp.sum(1 + log_var - mu**2 - jnp.exp(log_var / 2))

    return reconstruction_loss + kl_div


@dataclass
class LogMetrics:
    examples: list
    hiddens: Array
    labels: Array
    loss: float = 0

    def __init__(self, loader_length) -> None:
        self._hiddens_by_batch = []
        self._labels_by_batch = []
        self.__collect_labels = True
        self.num_batches = loader_length
        self.mykey = jr.key(1010)

    def reset(self, all=False):
        self.loss = 0
        self.examples = []
        self._hiddens_by_batch = []
        if all:
            self._labels_by_batch = []
            self.__collect_labels = True

    def log(self, loss, model, true_img, labels, save_plots=False):
        self.loss += loss / self.num_batches

        if save_plots:
            keys = jr.split(self.mykey, true_img.shape[0] + 1)
            self.mykey = keys[0]
            pred, hidden, *_ = jax.vmap(model)(keys[1:], true_img)

            id = np.random.randint(0, true_img.shape[0])
            self.examples.append((pred[id], true_img[id], labels[id]))
            self._hiddens_by_batch.append(hidden)

            if self.__collect_labels:
                self._labels_by_batch.append(labels)
            if len(self._labels_by_batch) == self.num_batches:
                self.__collect_labels = False

    def plot_latent_space(self):
        self.hiddens = jnp.concat(self._hiddens_by_batch)
        self.labels = jnp.concat(self._labels_by_batch)

        embeddings = TSNE().fit_transform(self.hiddens)  # pyright: ignore

        fig, ax = plt.subplots(constrained_layout=True)

        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=self.labels)
        return fig

    def plot_examples(self, num_examples=6):
        ids = np.random.randint(0, len(self.examples), num_examples)

        fig, axes = plt.subplots(
            2, num_examples, constrained_layout=True, figsize=(12, 6)
        )

        for i in range(num_examples):
            pred, true_img, label = self.examples[ids[i]]
            axes[0, i].imshow(true_img[0], cmap="gray")
            axes[0, i].set_title(label)
            axes[1, i].imshow(pred[0], cmap="gray")

        return fig


def train(
    model: MalariaAutoencoder | MalariaVAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optim: optax.GradientTransformation,
    loss_fn,
    nepochs: int,
) -> MalariaAutoencoder | MalariaVAE:
    train_metrics = LogMetrics(len(train_loader))
    test_metrics = LogMetrics(len(test_loader))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(model, opt_state, key, imgs: Float[Array, "batch 1 128 128"]):
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, key, imgs)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    key = jr.key(1026)

    for epoch in range(nepochs):
        log_summary = epoch % 2 == 0 or epoch == nepochs - 1
        train_metrics.reset()
        test_metrics.reset()

        for imgs, labels in train_loader:
            keys = jr.split(key, imgs.shape[0] + 1)
            key = keys[0]
            imgs = imgs.numpy()
            labels = labels.numpy()

            model, opt_state, loss_value = make_step(model, opt_state, keys[1:], imgs)

            train_metrics.log(
                loss_value.item(), model, imgs, labels, save_plots=log_summary
            )

        for imgs, labels in test_loader:
            keys = jr.split(key, imgs.shape[0] + 1)
            key = keys[0]
            imgs = imgs.numpy()
            labels = labels.numpy()
            loss_value = loss_fn(model, keys[1:], imgs)

            test_metrics.log(
                loss_value.item(), model, imgs, labels, save_plots=log_summary
            )

        log: dict[str, Any] = {
            "train_loss": train_metrics.loss,
            "test_loss": test_metrics.loss,
        }
        if log_summary:
            print(
                f"{epoch}: train loss: {train_metrics.loss:.4f}\t "
                f"test loss: {test_metrics.loss:.4f}"
            )

            log.update(
                {
                    "latent_train": wandb.Image(train_metrics.plot_latent_space()),
                    "latent_test": wandb.Image(test_metrics.plot_latent_space()),
                    "train_examples": wandb.Image(train_metrics.plot_examples()),
                    "test_examples": wandb.Image(test_metrics.plot_examples()),
                }
            )
            plt.close("all")
        wandb.log(log)

    wandb.finish()

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()
    return args


models = {
    "MalariaAutoencoder": MalariaAutoencoder,
    "MalariaVAE": MalariaVAE,
}

loss_fns = {
    "MalariaAutoencoder": loss_ae,
    "MalariaVAE": loss_vae,
}

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

    train_data, test_data = load_dataset(
        "autoencoders/data/malaria", remove_boundary=hyperparams["remove_cell_boundary"]
    )
    train_loader, test_loader = get_dataloaders(
        train_data, test_data, hyperparams["batch_size"]
    )

    optim = optax.adamw(hyperparams["learning_rate"])
    model = models[config["model"]](key=jax.random.key(1234), **hyperparams["model"])
    loss_fn = loss_fns[config["model"]]

    train(model, train_loader, test_loader, optim, loss_fn, hyperparams["nepochs"])
