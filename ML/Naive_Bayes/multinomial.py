import numpy as np
from collections import Counter, defaultdict

# Step 1: Load dataset
n = int(input())
data = []
for _ in range(n):
    row = input().split(',')  # split by comma
    text = row[0].strip()
    label = int(row[1].strip())
    data.append([text, label])

data = np.array(data)


# Step 2: Split train/test
train_size = int(0.7 * n)
X_train = data[:train_size, 0]
y_train = data[:train_size, 1].astype(int)
X_test = data[train_size:, 0]
y_test = data[train_size:, 1].astype(int)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# Step 3: Build vocabulary & compute priors
classes = np.unique(y_train)
priors = {}
word_counts = {}   # word counts per class
total_words = {}   # total words per class

vocab = set()

for c in classes:
    X_c = X_train[y_train == c]
    priors[c] = len(X_c) / len(y_train)
    
    counts = Counter()
    total = 0
    for doc in X_c:
        words = doc.lower().split()
        counts.update(words)
        total += len(words)
        vocab.update(words)
    word_counts[c] = counts
    total_words[c] = total

vocab = list(vocab)
V = len(vocab)

print(f"Class 0 Prior: {priors[0]:.2f}")
print(f"Class 1 Prior: {priors[1]:.2f}")


# Step 4: Conditional probabilities with Laplace smoothing
def cond_prob(word, c):
    return (word_counts[c].get(word, 0) + 1) / (total_words[c] + V)


# Step 5: Predict function
def predict(X):
    y_pred = []
    for doc in X:
        words = doc.lower().split()
        posteriors = {}
        for c in classes:
            log_prob = np.log(priors[c])  # use log to avoid underflow
            for w in words:
                log_prob += np.log(cond_prob(w, c))
            posteriors[c] = log_prob
        y_pred.append(max(posteriors, key=posteriors.get))
    return np.array(y_pred)


# Step 6: Make predictions

y_pred = predict(X_test)
print("Predictions:", y_pred.tolist())
print("Actual:     ", y_test.tolist())


# Step 7: Evaluation metrics
TP = np.sum((y_pred==1) & (y_test==1))
TN = np.sum((y_pred==0) & (y_test==0))
FP = np.sum((y_pred==1) & (y_test==0))
FN = np.sum((y_pred==0) & (y_test==1))

accuracy = (TP + TN) / len(y_test)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Accuracy={accuracy:.2f}")
print(f"Precision={precision:.2f}")
print(f"Recall={recall:.2f}")
print(f"F1={f1:.2f}")
