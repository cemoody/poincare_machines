import os.path
import numpy as np
from sklearn.model_selection import train_test_split

import requests
from zipfile import ZipFile


# Download, unzip and read in the dataset
name = 'ml-1m.zip'
base = 'ml-1m'
if not os.path.exists(name):
    url = 'http://files.grouplens.org/datasets/movielens/' + name
    r = requests.get(url)
    with open(name, 'wb') as fh:
        fh.write(r.content)
    zip = ZipFile(name)
    zip.extractall()

# First col is user, 2nd is movie id, 3rd is rating
data = np.genfromtxt(base + '/ratings.dat', delimiter='::')
# print("WARNING: Subsetting data")
# data = data[::100, :]
user = data[:, 0].astype('int32')
item = data[:, 1].astype('int32')
ratg = data[:, 2].astype('float32')
n_features = user.max() + item.max() + 1

tu, vu, ti, vi, tr, vr = train_test_split(user, item, ratg, random_state=42)
