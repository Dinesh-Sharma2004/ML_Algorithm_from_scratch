import numpy as np
import pandas as pd

def load_configuration(config_file='params.txt'):
    with open(config_file, 'r') as f:
        all_lines = f.readlines()
        
        num_clusters_line = all_lines[0]
        num_clusters = int(num_clusters_line.split('=')[1].strip())

        max_iter_line = all_lines[1]
        max_iter = int(max_iter_line.split('=')[1].strip())
            
    return num_clusters, max_iter

def assign_labels(points, cluster_centers):
    num_points = points.shape[0]
    labels = np.empty(num_points, dtype=int)
    
    for idx in range(num_points):
        point = points[idx]
        
        distances_sq = np.sum((point - cluster_centers) ** 2, axis=1)
        labels[idx] = np.argmin(distances_sq)
        
    return labels

def fit_model(training_data, num_clusters, max_iter):
    np.random.seed(42)
    
    indices = np.random.choice(training_data.shape[0], num_clusters, replace=False)
    centers = training_data[indices]
    
    labels = None
    for _ in range(max_iter):
        labels = assign_labels(training_data, centers)
        
        new_centers = np.zeros_like(centers)
        for i in range(num_clusters):
            points_in_cluster = training_data[labels == i]
            
            if len(points_in_cluster) > 0:
                new_centers[i] = np.mean(points_in_cluster, axis=0)
        
        centers = new_centers
        
    return centers, labels

def compute_metrics(points, centers, labels):
    total_wcss = 0.0
    for i, point in enumerate(points):
        assigned_center = centers[labels[i]]
        distance_sq = np.sum((point - assigned_center) ** 2)
        total_wcss += distance_sq

    mean_j = total_wcss / len(points)
    return mean_j, total_wcss

def run_pipeline():
    num_clusters, max_iterations = load_configuration()

    train_points = pd.read_csv('train.csv', header=None).values
    val_points = pd.read_csv('val.csv', header=None).values
    test_points = pd.read_csv('test.csv', header=None).values

    final_centers, train_labels = fit_model(train_points, num_clusters, max_iterations)

    j_train, wcss_train = compute_metrics(train_points, final_centers, train_labels)

    val_labels = assign_labels(val_points, final_centers)
    j_val, wcss_val = compute_metrics(val_points, final_centers, val_labels)
    
    test_labels = assign_labels(test_points, final_centers)
    j_test, wcss_test = compute_metrics(test_points, final_centers, test_labels)
    center_strs = [f"({c[0]:.2f},{c[1]:.2f})" for c in final_centers]
    formatted_centers = f"[{';'.join(center_strs)}]"

    print(f"Train_J={j_train:.2f}")
    print(f"Train_WCSS={wcss_train:.2f}")
    print(f"Val_J={j_val:.2f}")
    print(f"Val_WCSS={wcss_val:.2f}")
    print(f"Test_J={j_test:.2f}")
    print(f"Test_WCSS={wcss_test:.2f}")
    print(f"Centroids={formatted_centers}")

if __name__ == "__main__":
    run_pipeline()