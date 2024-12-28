import jax.random as jr
import matplotlib.pyplot as plt
from data.satellite_images import load_dataset


data, _ = load_dataset("gan/data/satellite_images")

idx = jr.randint(jr.key(9321), (12,), 0, len(data))

fig, axes = plt.subplots(3, 4, constrained_layout=True)

label_map = {
    0: "cloudy",
    1: "desert",
    2: "green_area",
    3: "water",
}

for id, ax in zip(idx, axes.ravel()):
    img, label = data[id]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(label_map[label])
    ax.set_axis_off()

plt.show()
