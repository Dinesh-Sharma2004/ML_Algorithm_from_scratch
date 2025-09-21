import numpy as np

n=int(input())
data=[]
for i in range(n):
    row=list(input().split(','))
    for i in range(len(row)):
        if row[i].isdigit():
            row[i]=float(row[i])
    data.append(row)

data=np.array(data)

print(f"{data[:5,:]}")
print(f"{data.shape}")

for col in range(data.shape[1]):
    x_mean=data[:,col].mean()
    x_max=data[:,col].max()
    x_min=data[:,col].min()
    x_std=data[:,col].std()
    print(f"{x_max:.2f} {x_min:.2f} {x_mean:.2f} {x_std:.2f}")

X=data[:,:-1]
y=data[:,-1]
X_mean=X.mean()
X_std=X.std()
X=(X-X_mean)/X_std

X = np.hstack((np.ones((X.shape[0], 1)), X))

def sigmoid(z):
    return 1/(1+np.exp(-z))


def LogisticRegression(X,y,alpha,epochs):
    m=X.shape[0]
    theta=np.zeros(X.shape[1])
    for _ in range(epochs):
        y_hat = sigmoid(X @ theta)
        error=y_hat-y
        gradient=(1/m)*(X.T@error)
        theta-=alpha*gradient
    return theta

theta=LogisticRegression(X,y,0.01,1000)
print(theta)
y_pred=sigmoid(X@theta)
eps = 1e-15
y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
n=X.shape[0]
loss=(-1/n)*(np.sum(y*np.log(y_pred_clipped)+(1-y)*np.log(1-y_pred_clipped)))
print(loss)

def predict(X_new, theta, X_mean, X_std):
    X_new = np.array(X_new, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)
    # Standardize using training mean & std
    X_new_std = (X_new - X_mean) / X_std
    X_new_std = np.hstack((np.ones((X_new_std.shape[0], 1)), X_new_std))
    return sigmoid(X_new_std @ theta)


samples = [
    [72, 80, 11],
    [150, 118, 20]
]

for s in samples:
    prob = predict(s, theta, X_mean, X_std)[0]
    print(f"Predicted admission probability for {s}: {prob:.4f}")

y_pred_label = (y_pred >= 0.5).astype(int)
accuracy = (y_pred_label == y).mean()
print("Training Accuracy:", accuracy)