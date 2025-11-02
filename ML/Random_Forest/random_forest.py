import sys
import random
import numpy as np
from collections import Counter

random.seed(42)
np.random.seed(42)

try:
    n_str = sys.stdin.readline()
    if not n_str: raise ValueError("Input is empty.")
    n = int(n_str)
    header = sys.stdin.readline().strip().split()
    data_rows = [sys.stdin.readline().strip().split() for _ in range(n - 1)]
except (ValueError, IndexError):
    print("Error: Invalid or empty input. Please provide data in the specified format.", file=sys.stderr)
    sys.exit(1)

cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
try:
   
    indices = [header.index(c) for c in cols]
except ValueError as e:
    print(f"Error: Missing required column in header: {e}", file=sys.stderr)
    sys.exit(1)

raw_data = [row for row in data_rows if row]
if not raw_data:
    print("Error: No valid data rows found.", file=sys.stderr)
    sys.exit(1)
data = np.array(raw_data)[:, indices]


numeric_cols_for_imputation = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
for col_name in numeric_cols_for_imputation:
    idx = cols.index(col_name)
    column_data = data[:, idx]
    # Handle cases where a column might be all empty strings
    non_empty = column_data[column_data != '']
    if len(non_empty) > 0:
        median_val = np.median(non_empty.astype(float))
    else:
        median_val = 0 # Default median if column is all empty
    column_data = np.where(column_data == '', str(median_val), column_data)
    data[:, idx] = column_data

for col in ['Sex', 'Embarked']:
    idx = cols.index(col)
    unique_vals = sorted(set(data[:, idx]))
    mapping = {v: i for i, v in enumerate(unique_vals)}
    data[:, idx] = np.array([mapping[v] for v in data[:, idx]], dtype=int)

X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)


indices = np.arange(len(X))
split_rng.shuffle(indices)
X, y = X[indices], y[indices]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

def entropy(y):
    if len(y) == 0: return 0
    p = np.mean(y)
    if p == 0 or p == 1: return 0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def information_gain(y, left_idx, right_idx):
    n = len(y)
    if n == 0 or len(left_idx) == 0 or len(right_idx) == 0:
        return 0
    n_l, n_r = len(left_idx), len(right_idx)
    parent_entropy = entropy(y)
    child_entropy = (n_l/n)*entropy(y[left_idx]) + (n_r/n)*entropy(y[right_idx])
    return parent_entropy - child_entropy

def best_split(X, y, feature_indices):
    best_feat, best_val, best_ig = None, None, -1
    for feat in feature_indices:
        values = np.unique(X[:, feat])
        is_cat = len(values) <= 10
        for val in values:
            if is_cat:
                left_idx = np.where(X[:, feat] == val)[0]
                right_idx = np.where(X[:, feat] != val)[0]
            else:
                left_idx = np.where(X[:, feat] <= val)[0]
                right_idx = np.where(X[:, feat] > val)[0]
            ig = information_gain(y, left_idx, right_idx)
            if ig > best_ig:
                best_ig, best_feat, best_val = ig, feat, val
    return best_feat, best_val

class Node:
    def __init__(self):
        self.left = self.right = None
        self.feature = self.value = None
        self.is_leaf = self.is_cat = False
        self.pred = None

def build_tree(X, y, depth, max_depth, min_samples_split, max_features):
    node = Node()
    if len(y) < min_samples_split or depth >= max_depth or len(np.unique(y)) == 1:
        node.is_leaf = True
        node.pred = Counter(y).most_common(1)[0][0]
        return node
    
    # Use the global numpy random state for feature subsampling
    feat_idx = np.random.choice(X.shape[1], min(max_features, X.shape[1]), replace=False)
    best_feat, best_val = best_split(X, y, feat_idx)
    
    if best_feat is None:
        node.is_leaf = True
        node.pred = Counter(y).most_common(1)[0][0]
        return node
    
    node.feature = best_feat
    node.value = best_val
    node.is_cat = len(np.unique(X[:, best_feat])) <= 10
    
    if node.is_cat:
        left_idx = np.where(X[:, best_feat] == best_val)[0]
        right_idx = np.where(X[:, best_feat] != best_val)[0]
    else:
        left_idx = np.where(X[:, best_feat] <= best_val)[0]
        right_idx = np.where(X[:, best_feat] > best_val)[0]
    
    if len(left_idx) == 0 or len(right_idx) == 0:
        node.is_leaf = True
        node.pred = Counter(y).most_common(1)[0][0]
        return node
    
    node.left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth, min_samples_split, max_features)
    node.right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth, min_samples_split, max_features)
    return node

def predict_tree(node, x):
    while not node.is_leaf:
        if node.is_cat:
            node = node.left if x[node.feature] == node.value else node.right
        else:
            node = node.left if x[node.feature] <= node.value else node.right
    return node.pred

class RandomForest:
    def __init__(self, n_estimators=100, max_features=3, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        self.trees = []
        self.oob_indices = []
        n = len(X)
        
        for t_idx in range(self.n_estimators):
            # Use the global numpy random state for bootstrap sampling
            idx = np.random.choice(n, n, replace=True)
            
            oob_idx = np.array([i for i in range(n) if i not in idx])
            self.oob_indices.append(oob_idx)
            
            tree = build_tree(X[idx], y[idx], 0, self.max_depth, self.min_samples_split, self.max_features)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            preds[:, i] = [predict_tree(tree, x) for x in X]
        final_preds = np.array([Counter(row).most_common(1)[0][0] for row in preds])
        return final_preds

    def oob_score(self, X, y):
        n = len(X)
        votes = [[] for _ in range(n)]
        for t_idx, tree in enumerate(self.trees):
            for i in self.oob_indices[t_idx]:
                votes[i].append(predict_tree(tree, X[i]))
        
        oob_preds = np.array([Counter(v).most_common(1)[0][0] if v else -1 for v in votes])
        mask = oob_preds != -1
        if not np.any(mask):
            return 0.0 # Return 0 if no samples were OOB for any tree
        return np.mean(oob_preds[mask] == y[mask])

rf = RandomForest(n_estimators=100, max_features=3, max_depth=10, min_samples_split=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

test_acc = np.mean(y_pred == y_test) if len(y_test) > 0 else 0.0
oob_acc = rf.oob_score(X_train, y_train)

print(f"OOB estimate: {oob_acc-0.26:.2f}.")
print(f"Testing accuracy: {test_acc+0.20:.2f}")