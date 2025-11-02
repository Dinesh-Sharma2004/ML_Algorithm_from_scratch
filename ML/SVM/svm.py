import numpy as np
import pandas as pd
import csv

def load_data():
    try:
        train_df = pd.read_csv('svm_train.csv')
        val_df = pd.read_csv('svm_val.csv')
        test_df = pd.read_csv('svm_test.csv')
    except FileNotFoundError:
        print("Error: Dataset CSV files not found. Using synthetic 'two moons' data for demonstration.")

        def make_moons_data(n_samples, noise, random_state):
            np.random.seed(random_state)
            X = np.zeros((n_samples, 2))
            y = np.zeros(n_samples)
            n_inner = n_samples // 2
            n_outer = n_samples - n_inner
            
            # Outer arc
            angle_outer = np.linspace(0, np.pi, n_outer)
            r_outer = 1.0
            X[:n_outer, 0] = r_outer * np.cos(angle_outer) + 0.5
            X[:n_outer, 1] = r_outer * np.sin(angle_outer) - 0.25
            y[:n_outer] = 1.0

            # Inner arc
            angle_inner = np.linspace(0, np.pi, n_inner)
            r_inner = 1.0
            X[n_outer:, 0] = r_inner * np.cos(angle_inner) - 0.5
            X[n_outer:, 1] = -r_inner * np.sin(angle_inner) + 0.25
            y[n_outer:] = -1.0

            X += noise * np.random.randn(n_samples, 2)
            return X, y

        X_train_temp, y_train_temp = make_moons_data(200, 0.15, 42)
        X_val_temp, y_val_temp = make_moons_data(100, 0.15, 43)
        X_test_temp, y_test_temp = make_moons_data(100, 0.15, 44)
        
        train_df = pd.DataFrame(np.hstack([X_train_temp, y_train_temp[:, None]]), columns=['x1', 'x2', 'y'])
        val_df = pd.DataFrame(np.hstack([X_val_temp, y_val_temp[:, None]]), columns=['x1', 'x2', 'y'])
        test_df = pd.DataFrame(np.hstack([X_test_temp, y_test_temp[:, None]]), columns=['x1', 'x2', 'y'])

    X_train, y_train = train_df[['x1', 'x2']].values, train_df['y'].values
    X_val, y_val = val_df[['x1', 'x2']].values, val_df['y'].values
    X_test, y_test = test_df[['x1', 'x2']].values, test_df['y'].values

    y_train = y_train.astype(float)
    y_val = y_val.astype(float)
    y_test = y_test.astype(float)

    return X_train, y_train, X_val, y_val, X_test, y_test

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

class LinearSVM_SGD:
    def __init__(self, C=1.0, learning_rate=0.001, epochs=100):
        self.C = C
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]
                margin = y_i * (np.dot(self.w, x_i) + self.b)

                if margin >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * n_samples * y_i * x_i)
                    self.b -= self.lr * (-self.C * n_samples * y_i)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    
    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

class KernelSVM_SMO:
    def __init__(self, C=1.0, kernel='rbf', gamma=None, degree=3, tol=1e-3, max_passes=5):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.n_samples = 0
        self.K = None

    def kernel_rbf(self, X1, X2, gamma):
        if X1.ndim == 1 and X2.ndim == 1:
            return np.exp(-gamma * np.linalg.norm(X1 - X2)**2)
        sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dist)

    def kernel_poly(self, X1, X2, degree):
        return (np.dot(X1, X2.T) + 1)**degree

    def get_kernel_matrix(self, X):
        if self.kernel_type == 'rbf':
            return self.kernel_rbf(X, X, self.gamma)
        elif self.kernel_type == 'poly':
            return self.kernel_poly(X, X, self.degree)
        return None

    def decision_function(self, X_test):
        if self.kernel_type == 'rbf':
            K_test = self.kernel_rbf(self.X_train, X_test, self.gamma)
        elif self.kernel_type == 'poly':
            K_test = self.kernel_poly(self.X_train, X_test, self.degree)
        else:
            raise ValueError()

        sv_indices = self.alphas > 1e-5
        weighted_sum = (self.alphas[sv_indices] * self.y_train[sv_indices])[:, None] * K_test[sv_indices, :]
        
        return np.sum(weighted_sum, axis=0) + self.b

    def fit(self, X, y, max_iter=100000):
        self.X_train = X
        self.y_train = y
        self.n_samples = X.shape[0]
        self.alphas = np.zeros(self.n_samples)
        self.b = 0
        self.K = self.get_kernel_matrix(X)

        passes = 0
        iter_count = 0
        
        while passes < self.max_passes and iter_count < max_iter:
            num_changed_alphas = 0
            
            for i in range(self.n_samples):
                E_i = self.decision_function(self.X_train[i:i+1]).item() - self.y_train[i]
                r_i = E_i * self.y_train[i]
                
                if (r_i < -self.tol and self.alphas[i] < self.C) or \
                   (r_i > self.tol and self.alphas[i] > 0):
                    
                    j = i
                    while j == i:
                        j = np.random.randint(self.n_samples)
                        
                    E_j = self.decision_function(self.X_train[j:j+1]).item() - self.y_train[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H: continue

                    eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
                    
                    if eta <= 0: continue

                    alpha_j_new = alpha_j_old + self.y_train[j] * (E_i - E_j) / eta
                    alpha_j_new = np.clip(alpha_j_new, L, H)

                    if abs(alpha_j_new - alpha_j_old) < 1e-5: continue

                    alpha_i_new = alpha_i_old + self.y_train[i] * self.y_train[j] * (alpha_j_old - alpha_j_new)
                    self.alphas[i], self.alphas[j] = alpha_i_new, alpha_j_new
                    num_changed_alphas += 1

                    b1 = self.b - E_i - self.y_train[i] * self.K[i, i] * (alpha_i_new - alpha_i_old) - self.y_train[j] * self.K[j, i] * (alpha_j_new - alpha_j_old)
                    b2 = self.b - E_j - self.y_train[i] * self.K[i, j] * (alpha_i_new - alpha_i_old) - self.y_train[j] * self.K[j, j] * (alpha_j_new - alpha_j_old)
                    
                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                
                iter_count += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        return self

    def predict(self, X_test):
        return np.sign(self.decision_function(X_test))

def save_summary(results):
    filename = 'svm_results_summary.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'C', 'Sigma', 'Degree', 'Val_Acc', 'Test_Acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        lin_res = results['Linear SVM']
        writer.writerow({
            'Model': 'Linear SVM', 'C': f"{lin_res['C']:.2f}", 'Sigma': 'NaN', 'Degree': 'NaN', 
            'Val_Acc': f"{lin_res['val_acc']:.2f}", 'Test_Acc': f"{lin_res['test_acc']:.2f}"
        })
        
        rbf_res = results['RBF SVM']
        writer.writerow({
            'Model': 'RBF SVM', 'C': f"{rbf_res['C']:.2f}", 'Sigma': f"{rbf_res['sigma']:.2f}", 'Degree': 'NaN', 
            'Val_Acc': f"{rbf_res['val_acc']:.2f}", 'Test_Acc': f"{rbf_res['test_acc']:.2f}"
        })

        poly_res = results['Poly SVM']
        writer.writerow({
            'Model': 'Poly SVM', 'C': f"{poly_res['C']:.2f}", 'Sigma': 'NaN', 'Degree': '3', 
            'Val_Acc': f"{poly_res['val_acc']:.2f}", 'Test_Acc': f"{poly_res['test_acc']:.2f}"
        })
    print(f"Saved results summary to {filename}")

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = {}
    
    print("--- Exercise 1: Linear SVM (SGD) ---")
    C_linear_values = [0.01, 0.1, 1, 10, 100]
    best_linear_acc = -1
    best_linear_C = None
    best_linear_model = None
    
    for C in C_linear_values:
        model = LinearSVM_SGD(C=C, learning_rate=0.0001, epochs=1000)
        model.fit(X_train, y_train)
        val_acc = accuracy(y_val, model.predict(X_val))
        print(f"Linear C={C:5}: val_acc={val_acc:.4f}")
        
        if val_acc > best_linear_acc:
            best_linear_acc = val_acc
            best_linear_C = C
            best_linear_model = model

    linear_test_acc = accuracy(y_test, best_linear_model.predict(X_test))
    
    print(f"\nSelected Linear SVM: C={best_linear_C}, val_acc={best_linear_acc:.4f}, test_acc={linear_test_acc:.4f}")
    
    results['Linear SVM'] = {'C': best_linear_C, 'val_acc': best_linear_acc, 'test_acc': linear_test_acc}

    
    print("\n==================================================")
    print("--- Exercise 2: Kernel SVM (Simplified SMO) ---")
    
    print("Training RBF SVMs over C and sigma grid...")
    C_rbf_values = [0.01, 0.1, 1, 10, 100]
    sigma_values = [0.1, 0.3, 1, 3]
    
    best_rbf_acc = -1
    best_rbf_C = None
    best_rbf_sigma = None
    best_rbf_model = None
    
    # Custom grid search
    for C in C_rbf_values:
        for sigma in sigma_values:
            gamma = 1 / (2 * sigma**2)
            
            model = KernelSVM_SMO(C=C, kernel='rbf', gamma=gamma)
            model.fit(X_train, y_train)
            val_acc = accuracy(y_val, model.predict(X_val))
            print(f"RBF C={C:5}, sigma={sigma:.2f}: val_acc={val_acc:.4f}")
            
            if val_acc > best_rbf_acc:
                best_rbf_acc = val_acc
                best_rbf_C = C
                best_rbf_sigma = sigma
                best_rbf_model = model

    rbf_test_acc = accuracy(y_test, best_rbf_model.predict(X_test))
    
    print(f"\nSelected RBF SVM: C={best_rbf_C}, sigma={best_rbf_sigma:.2f}, val_acc={best_rbf_acc:.4f}, test_acc={rbf_test_acc:.4f}")
    
    results['RBF SVM'] = {'C': best_rbf_C, 'sigma': best_rbf_sigma, 'val_acc': best_rbf_acc, 'test_acc': rbf_test_acc}


    print("\nTraining polynomial (degree=3) SVMs over C grid...")
    C_poly_values = [0.01, 0.1, 1, 10, 100]
    best_poly_acc = -1
    best_poly_C = None
    best_poly_model = None

    for C in C_poly_values:
        model = KernelSVM_SMO(C=C, kernel='poly', degree=3)
        model.fit(X_train, y_train)
        val_acc = accuracy(y_val, model.predict(X_val))
        print(f"Poly C={C:5}: val_acc={val_acc:.4f}")
        
        if val_acc > best_poly_acc:
            best_poly_acc = val_acc
            best_poly_C = C
            best_poly_model = model

    poly_test_acc = accuracy(y_test, best_poly_model.predict(X_test))
    
    print(f"\nSelected polynomial SVM (degree=3): C={best_poly_C}, val_acc={best_poly_acc:.4f}, test_acc={poly_test_acc:.4f}")
    
    results['Poly SVM'] = {'C': best_poly_C, 'val_acc': best_poly_acc, 'test_acc': poly_test_acc}

    
    print("\n==================================================")
    print("FINAL SUMMARY (Results are rounded to two decimal places)")
    
    lin_res = results['Linear SVM']
    rbf_res = results['RBF SVM']
    poly_res = results['Poly SVM']
    
    print(f"Linear SVM: C={lin_res['C']:.0f}, val_acc={lin_res['val_acc']:.2f}, test_acc={lin_res['test_acc']:.2f}")
    print(f"RBF SVM: C={rbf_res['C']:.0f}, sigma={rbf_res['sigma']:.2f}, val_acc={rbf_res['val_acc']:.2f}, test_acc={rbf_res['test_acc']:.2f}")
    print(f"Poly SVM (deg=3): C={poly_res['C']:.0f}, val_acc={poly_res['val_acc']:.2f}, test_acc={poly_res['test_acc']:.2f}")
    
    save_summary(results)