import os
from os.path import dirname, join, normpath
import pandas as pd
import sys

def _load_data(csv_data):
    BASE_DIR = normpath(dirname("__file__"))
    data = pd.read_csv(join(BASE_DIR, csv_data))
    return data

def texts_data(csv_data):
    data = _load_data(csv_data)
    texts = data["text"]
    return texts

def labels_data(csv_data):
    data = _load_data(csv_data)
    labels = data["label"]
    return labels

def prepare_data(csv_data):
    data = _load_data(csv_data)
    texts = data["text"]
    labels = data["label"]
    return texts, labels
