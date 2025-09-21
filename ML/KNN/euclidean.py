import numpy as np
from collections import Counter


n = int(input())  # number of rows
data = []
for _ in range(n):
    row = input().split()
    features = list(map(float, row[:4]))  
    label = row[4]                       
    data.append(features + [label])

data = np.array(data, dtype=object)

X = data[:, :4].astype(float)  # features
y = data[:, 4]                 # labels


np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))  # vectorized for multiple training points

def knn_predict(X_train, y_train, x_test, k):
    distances = euclidean_distance(X_train, x_test)
    neighbor_idx = np.argsort(distances)[:k]   # indices of k nearest
    neighbor_labels = y_train[neighbor_idx]
    most_common = Counter(neighbor_labels).most_common(1)[0][0]
    return most_common


def cross_val_knn(X_train, y_train, k, folds=5):
    fold_size = len(X_train) // folds
    accuracies = []

    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < folds - 1 else len(X_train)

        X_val = X_train[start:end]
        y_val = y_train[start:end]
        X_tr = np.vstack((X_train[:start], X_train[end:]))
        y_tr = np.hstack((y_train[:start], y_train[end:]))

        correct = 0
        for j in range(len(X_val)):
            pred = knn_predict(X_tr, y_tr, X_val[j], k)
            if pred == y_val[j]:
                correct += 1
        accuracies.append(correct / len(X_val))

    return np.mean(accuracies)


ks = [1, 3, 5, 7, 9]
print(" k | CV Accuracy")
for k in ks:
    acc = cross_val_knn(X_train, y_train, k)
    print(f"{k:2d} | {acc*100:.2f}%")
