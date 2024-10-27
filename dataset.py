import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

cifar10 = fetch_openml('cifar_10', version=1, cache=True)

X = cifar10.data
y = cifar10.target

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0