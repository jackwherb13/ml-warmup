import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from mnist import load_images, load_labels, prepare

DATA = Path(__file__).parent.parent / "data"


def write_labels(path, labels, magic=2049):
    # Build a tiny IDX label file: 8-byte big-endian header, then one byte per label
    header = struct.pack(">II", magic, len(labels))
    with gzip.open(path, "wb") as f:
        f.write(header + bytes(labels))


def write_images(path, pixels, magic=2051):
    # Build a tiny IDX image file: 16-byte big-endian header, then one byte per pixel
    count, rows, cols = pixels.shape
    header = struct.pack(">IIII", magic, count, rows, cols)
    with gzip.open(path, "wb") as f:
        f.write(header + pixels.astype(np.uint8).tobytes())


def test_load_labels_reads_values(tmp_path):
    path = tmp_path / "labels.gz"
    write_labels(path, [7, 2, 1, 0, 9])

    labels = load_labels(path)

    assert labels.shape == (5,)
    assert labels.dtype == np.uint8
    assert labels.tolist() == [7, 2, 1, 0, 9]


def test_load_labels_rejects_wrong_magic(tmp_path):
    path = tmp_path / "images.gz"
    write_labels(path, [1, 2, 3], magic=2051)

    with pytest.raises(ValueError):
        load_labels(path)


@pytest.mark.parametrize("name, count", [
    ("train-labels-idx1-ubyte.gz", 60000),
    ("t10k-labels-idx1-ubyte.gz", 10000),
])
def test_load_labels_real_files(name, count):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not downloaded")

    labels = load_labels(path)

    assert labels.shape == (count,)
    assert labels.min() == 0
    assert labels.max() == 9


def test_load_images_reads_pixels(tmp_path):
    path = tmp_path / "images.gz"
    pixels = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    write_images(path, pixels)

    images = load_images(path)

    assert images.shape == (2, 3, 4)
    assert images.dtype == np.uint8
    assert np.array_equal(images, pixels)


def test_load_images_rejects_wrong_magic(tmp_path):
    path = tmp_path / "labels.gz"
    write_images(path, np.zeros((1, 2, 2)), magic=2049)

    with pytest.raises(ValueError):
        load_images(path)


@pytest.mark.parametrize("name, count", [
    ("train-images-idx3-ubyte.gz", 60000),
    ("t10k-images-idx3-ubyte.gz", 10000),
])
def test_load_images_real_files(name, count):
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} not downloaded")

    images = load_images(path)

    assert images.shape == (count, 28, 28)
    assert images.dtype == np.uint8


def fake_images():
    # 2 images of 3x4, covering the full 0..255 range
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    images[0, 0, 0] = 0
    images[1, 2, 3] = 255
    return images


def test_prepare_flattens_each_image_to_a_row():
    out = prepare(fake_images())

    assert out.shape == (2, 12)


def test_prepare_returns_floats():
    out = prepare(fake_images())

    assert np.issubdtype(out.dtype, np.floating)


def test_prepare_scales_to_0_1():
    out = prepare(fake_images())

    assert out.min() == 0.0
    assert out.max() == 1.0


def test_prepare_keeps_pixel_order():
    images = fake_images()

    out = prepare(images)

    # row i is image i read left to right, top to bottom
    assert np.allclose(out[0], images[0].ravel() / 255)
    assert np.allclose(out[1], images[1].ravel() / 255)
