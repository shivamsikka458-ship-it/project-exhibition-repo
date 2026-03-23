import numpy as np
import pandas as pd

df = pd.read_csv("MP_Data.csv")

X, y = [], []

for i in range(len(df)):
    file = df.iloc[i]['filename']
    label = df.iloc[i]['label']
    
    arr = np.load("data/" + file)
    X.append(arr)
    y.append(label)

print("Dataset Loaded")
print("X shape:", len(X))
print("y shape:", len(y))