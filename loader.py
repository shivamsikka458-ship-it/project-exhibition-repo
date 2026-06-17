import numpy as np
import os

data_path = "sign_language_dataset/data"

X = []
y = []

actions = set()

# get labels automatically
for file in os.listdir(data_path):
    if file.endswith(".npy"):
        actions.add(file.split("_")[0])

actions = list(actions)
label_map = {label: i for i, label in enumerate(actions)}

# load data
for file in os.listdir(data_path):
    if file.endswith(".npy"):
        sequence = np.load(os.path.join(data_path, file))
        X.append(sequence)

        word = file.split("_")[0]
        y.append(label_map[word])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)
