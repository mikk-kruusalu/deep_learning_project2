# Autoencoders, GANs and PINNs

This is another project from the Deep learning course in Taltech autumn 2024. For the other one see <https://github.com/mikk-kruusalu/deep_learning_project>.

In order to use this code, one should create a virtual environment with `python -m venv venv` and activate it. On Linux, the activation is done with `source venv/bin/activate`. The the required packages should be installed with

```bash
pip install -r requirements.txt
```

To add GPU acceleration for Jax, see the [install guide](https://jax.readthedocs.io/en/latest/installation.html). Also, [Weights & Biases](wandb.ai) account is needed for autoencoders and GANs, the login on the command line is done with `wandb login`. The wandb project can be configured via the `.yaml` config files.

Please find the reports for the course in the root of this repository.
