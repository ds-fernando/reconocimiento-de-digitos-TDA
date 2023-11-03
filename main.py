from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
from gtda.plotting import plot_heatmap
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser='auto')

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

im8_idx = np.flatnonzero(y == "8")[0]
img8 = X[im8_idx].reshape(28, 28)
plot_heatmap(img8)
     

train_size, test_size = 60, 10
X = X.reshape((-1, 28, 28))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=666
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
     