import numpy as np

# Read input
n = int(input())
data = [list(map(float, input().split())) for _ in range(n)]
data = np.array(data)
X = data[:, :-1]
y = data[:, -1]

# Train-test-val split
split_idx = int(n * 0.7)
val_idx = split_idx + (n - split_idx) // 2

X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:val_idx], y[split_idx:val_idx]
X_val, y_val = X[val_idx:], y[val_idx:]

# Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree
class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_leaf):
            return Node(value=self._most_common(y))

        feature, threshold = self._best_split(X, y, num_features)
        if feature is None:
            return Node(value=self._most_common(y))

        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) == 0 or len(right) == 0:
                    continue

                gain = self._information_gain(y, left, right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, parent, left, right):
        weight_l = len(left) / len(parent)
        weight_r = len(right) / len(parent)

        if self.criterion == 'gini':
            gain = self._gini(parent) - (weight_l * self._gini(left) + weight_r * self._gini(right))
        else:
            gain = self._entropy(parent) - (weight_l * self._entropy(left) + weight_r * self._entropy(right))

        return gain

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return -np.sum(prob * np.log2(prob + 1e-9))  # Small epsilon for log(0)

    def _most_common(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

# Accuracy function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Search over depth and min_leaf
depths = [2, 3, 4, 5, 6]
min_leafs = [1, 3, 5, 10]

best_entropy = (0, 0, 0.0, 0.0)  # depth, min_leaf, val_acc, test_acc
best_gini = (0, 0, 0.0, 0.0)

for depth in depths:
    for min_leaf in min_leafs:
        # Entropy
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, criterion='entropy')
        tree.fit(X_train, y_train)
        val_preds = tree.predict(X_val)
        val_acc = accuracy(y_val, val_preds)
        if val_acc > best_entropy[2]:
            test_preds = tree.predict(X_test)
            test_acc = accuracy(y_test, test_preds)
            best_entropy = (depth, min_leaf, val_acc, test_acc)

        # Gini
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, criterion='gini')
        tree.fit(X_train, y_train)
        val_preds = tree.predict(X_val)
        val_acc = accuracy(y_val, val_preds)
        if val_acc > best_gini[2]:
            test_preds = tree.predict(X_test)
            test_acc = accuracy(y_test, test_preds)
            best_gini = (depth, min_leaf, val_acc, test_acc)

# Output in required format
print(f"Best (entropy): depth={best_entropy[0]}, minleaf={best_entropy[1]}, val_acc={best_entropy[3]:.2f}, test_acc={best_entropy[2]:.2f}")
print(f"Best (gini):    depth={best_gini[0]}, minleaf={best_gini[1]}, val_acc={best_gini[3]:.2f}, test_acc={best_gini[2]:.2f}")