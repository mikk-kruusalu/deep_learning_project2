import argparse
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
            pred, hidden = jax.vmap(model)(true_img)

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

        fig, ax = plt.subplots(constrained_layout=True)

        ax.scatter(self.hiddens[:, 0], self.hiddens[:, 1], c=self.labels)
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
    model: MalariaAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optim: optax.GradientTransformation,
    nepochs: int,
) -> MalariaAutoencoder:
    train_metrics = LogMetrics(len(train_loader))
    test_metrics = LogMetrics(len(test_loader))
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
        log_summary = epoch % 2 == 0 or epoch == nepochs - 1
        train_metrics.reset()
        test_metrics.reset()

        for imgs, labels in train_loader:
            imgs = imgs.numpy()
            labels = labels.numpy()

            model, opt_state, loss_value = make_step(model, opt_state, imgs)

            train_metrics.log(
                loss_value.item(), model, imgs, labels, save_plots=log_summary
            )

        for imgs, labels in test_loader:
            imgs = imgs.numpy()
            labels = labels.numpy()
            loss_value = loss(model, imgs)

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
