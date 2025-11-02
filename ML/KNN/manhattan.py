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

X = data[:, :4].astype(float)
y = data[:, 4]


# 2. Train-Test Split (80-20)
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Manhattan Distance
def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)

def knn_predict(X_train, y_train, x_test, k):
    distances = manhattan_distance(X_train, x_test)
    neighbor_idx = np.argsort(distances)[:k]
    neighbor_labels = y_train[neighbor_idx]
    most_common = Counter(neighbor_labels).most_common(1)[0][0]
    return most_common

# 4. 5-Fold Cross-Validation
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


# 5. Evaluate for k = 1,3,5,7,9
ks = [1, 3, 5, 7, 9]
print(" k | CV Accuracy (Manhattan)")
for k in ks:
    acc = cross_val_knn(X_train, y_train, k)
    print(f"{k:2d} | {acc*100:.2f}%")

<<<<<<< HEAD

=======
>>>>>>> 504d6a194ad2396fb5a3d3f4b6b42e5121161da3
best_k = max(ks, key=lambda k: cross_val_knn(X_train, y_train, k))
print(f"\nBest k based on CV: {best_k}")

correct = 0
for i in range(len(X_test)):
    pred = knn_predict(X_train, y_train, X_test[i], best_k)
    if pred == y_test[i]:
        correct += 1
test_acc = correct / len(X_test)
<<<<<<< HEAD
print(f"Test Accuracy with k={best_k}: {test_acc*100:.2f}%")
=======
print(f"Test Accuracy with k={best_k}: {test_acc*100:.2f}%")

>>>>>>> 504d6a194ad2396fb5a3d3f4b6b42e5121161da3
