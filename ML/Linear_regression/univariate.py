import numpy as np

# Input number of rows (samples)
n = int(input("Enter number of samples: "))

data = []
for i in range(n):
    row = list(map(float, input().split()))  # directly convert all to float
    data.append(row)

# Convert to numpy array
data = np.array(data)

print("\nDataset:")
print(data)

print("\nShape of dataset (rows, cols):", data.shape)

# Basic statistics (for each column separately)
for col in range(data.shape[1]):
    print(f"\nColumn {col}: mean={data[:,col].mean():.2f}, std={data[:,col].std():.2f}, "
          f"min={data[:,col].min():.2f}, max={data[:,col].max():.2f}")

# Separate features (X) and target (y)
X = data[:, :-1]   # all columns except last
y = data[:, -1]    # last column = target

# Normalize features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
epsilon = 1e-8
X_scaled = (X - X_mean) / (X_std + epsilon)

# Add bias column (intercept term)
X_b = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))

def Linear_regression(X, y, alpha, epochs):
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    for i in range(epochs):
        y_hat = X @ theta
        error = y_hat - y
        gradient = (1/m) * X.T @ error
        theta -= alpha * gradient
    return theta

# Train model
theta = Linear_regression(X_b, y, alpha=0.01, epochs=1000)
print("\nLearned parameters (theta):", theta)

# Prediction function
def predict(X_new, theta, X_mean, X_std):
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)  # reshape single sample
    epsilon = 1e-8
    X_scaled = (X_new - X_mean) / (X_std + epsilon)
    X_b = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
    return X_b @ theta

# Training error
y_pred = X_b @ theta
MSE = np.mean((y_pred - y) ** 2) / 2
print("\nTraining MSE:", MSE)

# Test predictions
print("\n--- Predictions ---")
new_points = [
    np.array([150]),  
    np.array([200])   
]
for p in new_points:
    pred = predict(p, theta, X_mean, X_std)
    print(f"Predicted value for {p}: {pred[0]:.2f}")
