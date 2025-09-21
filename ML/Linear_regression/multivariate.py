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
    mse_history = []

    for i in range(epochs):
        y_hat = X @ theta
        error = y_hat - y
        gradient = (1/m) * X.T @ error
        theta -= alpha * gradient

        # track MSE (Mean Squared Error / 2)
        mse = np.mean((y_hat - y) ** 2) / 2
        mse_history.append(mse)

    return theta, mse_history

# Train model with Gradient Descent
theta_gd, mse_history = Linear_regression(X_b, y, alpha=0.01, epochs=300)

# Normal Equation solution: θ = (XᵀX)⁻¹ Xᵀy
theta_normal = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

# Predictions using GD and Normal Equation
y_pred_gd = X_b @ theta_gd
y_pred_normal = X_b @ theta_normal

# Compute MSE values
initial_MSE = mse_history[0]
final_MSE = mse_history[-1]
mse_diff_improvement = initial_MSE - final_MSE
mse_gd = np.mean((y_pred_gd - y) ** 2) / 2
mse_normal = np.mean((y_pred_normal - y) ** 2) / 2
mse_diff_gd_normal = abs(mse_gd - mse_normal)

print("\nLearned parameters (Gradient Descent):", theta_gd)
print("Learned parameters (Normal Equation):", theta_normal)


print(f"Final MSE={final_MSE:.2f}")
print(f"MSE Difference (GD vs Normal)={mse_diff_gd_normal:.5f}")

# Prediction function
def predict(X_new, theta, X_mean, X_std):
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)  # reshape single sample
    epsilon = 1e-8
    X_scaled = (X_new - X_mean) / (X_std + epsilon)
    X_b = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
    return X_b @ theta

print("\n--- Predictions ---")
new_points = [
    np.array([150, 3, 5]),
    np.array([200, 4, 2])
]
for p in new_points:
    pred_gd = predict(p, theta_gd, X_mean, X_std)
    pred_normal = predict(p, theta_normal, X_mean, X_std)
    print(f"Point {p} → GD Prediction={pred_gd[0]:.2f}, Normal Prediction={pred_normal[0]:.2f}")
