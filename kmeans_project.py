import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# -------------------------------
# STEP 1 : Create sample dataset
# -------------------------------
data, true_labels = make_blobs(n_samples=300, centers=5, random_state=10)


# -------------------------------
# STEP 2 : K-Means from scratch
# -------------------------------
def run_kmeans(points, k, max_iter=60):

    # randomly pick starting centers
    idx = np.random.choice(len(points), k, replace=False)
    centers = points[idx]

    for iteration in range(max_iter):

        # assign each point to nearest center
        cluster_id = []
        for p in points:
            dist = []
            for c in centers:
                d = np.linalg.norm(p - c)
                dist.append(d)

            cluster_id.append(np.argmin(dist))

        cluster_id = np.array(cluster_id)

        # recompute centers
        new_centers = []
        for i in range(k):
            members = points[cluster_id == i]

            if len(members) == 0:
                new_centers.append(centers[i])   # keep old center
            else:
                new_centers.append(members.mean(axis=0))

        new_centers = np.array(new_centers)

        # check convergence
        if np.allclose(centers, new_centers):
            print(f"Converged at iteration {iteration}")
            break

        centers = new_centers

    return cluster_id, centers


# -------------------------------
# STEP 3 : WCSS calculation
# -------------------------------
def find_wcss(points, labels, centers):
    total = 0
    for i in range(len(points)):
        center = centers[labels[i]]
        total += np.linalg.norm(points[i] - center) ** 2
    return total


# -------------------------------
# STEP 4 : Elbow method
# -------------------------------
scores = []

print("K vs WCSS values")
for k in range(1, 11):
    lbl, ctr = run_kmeans(data, k)
    value = find_wcss(data, lbl, ctr)
    scores.append(value)
    print(f"K = {k}  -->  WCSS = {value:.2f}")

# automatic elbow detection
diffs = np.diff(scores)
diff_diffs = np.diff(diffs)

optimal_k = np.agrmin(diff_diffs) + 2
print("Optimal K detected:", optimal_k)
# -----------------------------

# plot elbow
plt.plot(range(1, 11), scores, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method Graph")
plt.show()


# -------------------------------
# STEP 5 : Final clustering
# -------------------------------

labels, centers = run_kmeans(data, optimal_k)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)
plt.title("Final Cluster Output")
plt.show()
