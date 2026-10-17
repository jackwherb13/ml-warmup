# mnist.py
#Turns four dataset files into NumPy arrays

import gzip
import struct
import numpy as np


def load_images(path):
    # 1. open the .gz file in binary mode
    with gzip.open(path, 'rb') as f:
        header_bytes = f.read(16)
        
        # 2. read the 16-byte header and unpack it into 4 numbers
        magic, count, rows, cols = struct.unpack('>4I', header_bytes)

        # 3. if the magic number isn't 2051, raise an error
        if magic != 2051:
            raise ValueError(f"Expected 2051, was not found got {magic}")

        # 4. read the rest of the bytes as a NumPy array of single bytes
        pixel_bytes = f.read()

        # 5. reshape into (count, rows, cols) and return it
        pixels = np.frombuffer(pixel_bytes, dtype=np.uint8)

        return pixels.reshape(count, rows, cols)



def load_labels(path):
    # same idea: 8-byte header, magic number 2049, then one byte per label

    #open in binary mode
    with gzip.open(path, 'rb') as f:
        header_bytes = f.read(8)

        #unpack header into magic and count
        magic, count = struct.unpack('>2I', header_bytes)

        #if the magic number isn't 2049, raise an error
        if magic != 2049:
            raise ValueError(f"Expected 2049, was not found got {magic}")

        pixel_bytes = f.read()

        pixels = np.frombuffer(pixel_bytes, dtype=np.uint8)

    return pixels

def prepare(images):

    count, rows, cols = images.shape
    reshape = images.reshape(count, rows * cols)

    normalized = reshape / 255

    return normalized