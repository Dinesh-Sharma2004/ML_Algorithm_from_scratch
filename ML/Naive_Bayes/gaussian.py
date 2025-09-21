import numpy as np

# Step 1: Load dataset

n = 30  # number of rows
data = []
for _ in range(n):
    row = input().split()
    # Convert first 8 features to float, 9th is class, ignore last column
    features = list(map(float, row[:8]))
    label = int(row[8])
    data.append(features + [label])

data = np.array(data)  # shape (30, 9)


# Step 2: Split train/test
train_size = int(0.7 * n)
X_train = data[:train_size, :-1]
y_train = data[:train_size, -1].astype(int)
X_test = data[train_size:, :-1]
y_test = data[train_size:, -1].astype(int)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# Step 3: Compute priors and feature stats per class
classes = np.unique(y_train)
priors = {}
means = {}
stds = {}

for c in classes:
    X_c = X_train[y_train == c]
    priors[c] = len(X_c) / len(y_train)
    means[c] = X_c.mean(axis=0)      # mean per feature
    stds[c] = X_c.std(axis=0)        # std per feature

print(f"Class 0 Prior: {priors[0]:.2f}")
print(f"Class 1 Prior: {priors[1]:.2f}")


# Step 4: Gaussian PDF

def gaussian_pdf(x, mean, std):
    eps = 1e-8  # avoid division by zero
    coeff = 1.0 / np.sqrt(2 * np.pi * (std**2 + eps))
    exponent = -((x - mean)**2) / (2 * (std**2 + eps))
    return coeff * np.exp(exponent)


# Step 5: Predict function
def predict(X):
    y_pred = []
    for x in X:
        posteriors = {}
        for c in classes:
            likelihoods = gaussian_pdf(x, means[c], stds[c])
            posterior = np.prod(likelihoods) * priors[c]
            posteriors[c] = posterior
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
