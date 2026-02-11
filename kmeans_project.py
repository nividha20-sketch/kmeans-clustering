import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# -------------------------------
# STEP 1 : Generate dataset
# -------------------------------
data, true_labels = make_blobs(n_samples=300, centers=5, random_state=10)


# -------------------------------
# STEP 2 : K-Means from scratch
# -------------------------------
def run_kmeans(points, k, max_iter=100):

    # random initialization
    idx = np.random.choice(len(points), k, replace=False)
    centers = points[idx]

    for iteration in range(max_iter):

        # VECTORISED distance calculation (improved performance)
        distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        cluster_id = np.argmin(distances, axis=1)

        # recompute centers
        new_centers = []
        for i in range(k):
            members = points[cluster_id == i]
            if len(members) == 0:
                new_centers.append(centers[i])
            else:
                new_centers.append(members.mean(axis=0))

        new_centers = np.array(new_centers)

        # convergence check using centroid shift
        shift = np.linalg.norm(new_centers - centers)
        if shift < 1e-4:
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
# STEP 4 : Compute WCSS for K=1..10
# -------------------------------
scores = []

print("\n----- WCSS RESULTS -----")
for k in range(1, 11):
    lbl, ctr = run_kmeans(data, k)
    value = find_wcss(data, lbl, ctr)
    scores.append(value)
    print(f"K={k}: {value:.2f}")
print("------------------------\n")


# -------------------------------
# STEP 5 : Robust Elbow Detection
# Distance from line method
# -------------------------------
k_values = np.arange(1, len(scores)+1)

x1, y1 = 1, scores[0]
x2, y2 = len(scores), scores[-1]

distances = []
for i in range(len(scores)):
    x0 = k_values[i]
    y0 = scores[i]

    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distances.append(numerator/denominator)

optimal_k = np.argmax(distances) + 1
print("Optimal K detected:", optimal_k)


# -------------------------------
# STEP 6 : Plot elbow graph
# -------------------------------
plt.figure()
plt.plot(range(1, 11), scores, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method Graph")
plt.savefig("elbow.png")
plt.close()


# -------------------------------
# STEP 7 : Final clustering
# -------------------------------
labels, centers = run_kmeans(data, optimal_k)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)
plt.title("Final Cluster Output")
plt.savefig("clusters.png")
plt.close()

print("\nClustering complete. Graphs saved as elbow.png and clusters.png")
