import jax.random as jr
import matplotlib.pyplot as plt
from data.malaria import load_dataset


data, _ = load_dataset("autoencoders/data/malaria")

idx = jr.randint(jr.key(9321), (12,), 0, len(data))

fig, axes = plt.subplots(3, 4, constrained_layout=True)

label_map = {0: "infected", 1: "uninfected"}

for id, ax in zip(idx, axes.ravel()):
    img, label = data[id]
    ax.imshow(img.permute(1, 2, 0), cmap="gray")
    ax.set_title(label_map[label])
    ax.set_axis_off()

plt.show()
