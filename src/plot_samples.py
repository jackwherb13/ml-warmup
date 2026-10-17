import matplotlib.pyplot as plt
from pathlib import Path
from mnist import load_images, load_labels

DATA = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent.parent / "out"

OUT.mkdir(parents=True, exist_ok=True)

train_images = DATA / "train-images-idx3-ubyte.gz"
train_labels = DATA / "train-labels-idx1-ubyte.gz"

images = load_images(train_images)
labels = load_labels(train_labels)

fig, axes = plt.subplots(4, 4) 

for index, ax in enumerate(axes.flat):
    ax.imshow(images[index], cmap="gray")
    ax.set_title(labels[index])
    ax.axis("off")

fig.tight_layout()

fig.savefig(OUT / "samples.png")