import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from mnist import load_labels

DATA = Path(__file__).parent.parent / "data"


def write_labels(path, labels, magic=2049):
    # Build a tiny IDX label file: 8-byte big-endian header, then one byte per label
    header = struct.pack(">II", magic, len(labels))
    with gzip.open(path, "wb") as f:
        f.write(header + bytes(labels))


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
