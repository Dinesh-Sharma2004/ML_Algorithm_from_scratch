import numpy as np
import pandas as pd
import sys

class ClusterModel:
    def __init__(self, n_clusters, max_steps=100, seed_val=0):
        self.n_clusters = n_clusters
        self.max_steps = max_steps
        self.seed_val = seed_val
        self.centers = None

    def _init_centers_plus(self, data):
        chosen = [data[np.random.choice(data.shape[0])]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in chosen]) for x in data])
            prob = dist_sq / dist_sq.sum()
            csum = prob.cumsum()
            rand_val = np.random.rand()
            for idx, p in enumerate(csum):
                if rand_val < p:
                    chosen_idx = idx
                    break
            chosen.append(data[chosen_idx])
        self.centers = np.array(chosen)

    def _assign_points(self, data):
        dists = np.sqrt(((data - self.centers[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(dists, axis=0)

    def _recompute_centers(self, data, labels):
        new_centers = np.zeros((self.n_clusters, data.shape[1]))
        for i in range(self.n_clusters):
            members = data[labels == i]
            if len(members) > 0:
                new_centers[i] = np.mean(members, axis=0)
            else:
                new_centers[i] = data[np.random.choice(data.shape[0])]
        return new_centers

    def train(self, data):
        np.random.seed(self.seed_val)
        self._init_centers_plus(data)
        prev_labels = None
        for _ in range(self.max_steps):
            labels = self._assign_points(data)
            if prev_labels is not None and np.array_equal(prev_labels, labels):
                break
            prev_labels = labels
            self.centers = self._recompute_centers(data, labels)

    def infer(self, data):
        return self._assign_points(data)

    def _compute_wcss(self, data, labels):
        wcss_score = 0
        for i in range(self.n_clusters):
            members = data[labels == i]
            if len(members) > 0:
                wcss_score += np.sum((members - self.centers[i]) ** 2)
        return wcss_score


def main():
    k_line = sys.stdin.readline()
    iter_line = sys.stdin.readline()
    k_limit = int(k_line.split('=')[1])
    n_iter = int(iter_line.split('=')[1])

    data_train = pd.read_csv('train.csv', header=None).values
    data_val = pd.read_csv('val.csv', header=None).values
    data_test = pd.read_csv('test.csv', header=None).values

    wcss_vals = []
    for k in range(1, k_limit + 1):
        model = ClusterModel(n_clusters=k, max_steps=n_iter, seed_val=0)
        model.train(data_train)
        val_labels = model.infer(data_val)
        wcss_vals.append(model._compute_wcss(data_val, val_labels))

    all_pts = np.array([[k, w] for k, w in enumerate(wcss_vals, 1)])
    start_pt, end_pt = all_pts[0], all_pts[-1]

    if k_limit > 1:
        line_vec = end_pt - start_pt
        norm = np.sqrt(np.sum(line_vec ** 2))
        if norm == 0:
            dists = np.sqrt(np.sum((all_pts - start_pt) ** 2, axis=1))
        else:
            diff = all_pts - start_pt
            cross = np.cross(diff, line_vec)
            dists = np.abs(cross) / norm
        elbow_idx = np.argmax(dists)
        elbow_k = int(all_pts[elbow_idx, 0])
    else:
        elbow_k = 1

    final_model = ClusterModel(n_clusters=elbow_k, max_steps=n_iter, seed_val=0)
    final_model.train(data_train)
    test_labels = final_model.infer(data_test)
    test_wcss = final_model._compute_wcss(data_test, test_labels)
    test_j = test_wcss / len(data_test)

    print(f"Elbow_K={elbow_k}")
    print(f"WCSS_Val=[{','.join([f'{w:.2f}' for w in wcss_vals])}]")
    print(f"Test_J_Elbow={test_j:.2f} Test_WCSS_Elbow={test_wcss:.2f}")


if __name__ == "__main__":
    main()
