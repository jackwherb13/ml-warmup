from pathlib import Path
import urllib.request

url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

DATA = Path(__file__).parent.parent / "data"


t10k_images =  DATA / Path("t10k-images-idx3-ubyte.gz")
t10k_labels = DATA / Path("t10k-labels-idx1-ubyte.gz")
train_images = DATA / Path("train-images-idx3-ubyte.gz")
train_labels = DATA / Path("train-labels-idx1-ubyte.gz")

DATA.mkdir(parents=True, exist_ok=True)

if not t10k_images.is_file():
    print("No file found for t10k_images")
    urllib.request.urlretrieve(url + "t10k-images-idx3-ubyte.gz", t10k_images)
    print("Download complete")
else:
    print("File found")

if not t10k_labels.is_file():
    print("No file found for t10k_labels")
    urllib.request.urlretrieve(url + "t10k-labels-idx1-ubyte.gz", t10k_labels)
    print("Download complete")
else:
    print("File found")

if not train_images.is_file():
    print("No file found for train images")
    urllib.request.urlretrieve(url + "train-images-idx3-ubyte.gz", train_images)
    print("Download complete")
else:
    print("File found")

if not train_labels.is_file():
    print("No file found for train labels")
    urllib.request.urlretrieve(url + "train-labels-idx1-ubyte.gz", train_labels)
    print("Download complete")
else:
    print("File found")
