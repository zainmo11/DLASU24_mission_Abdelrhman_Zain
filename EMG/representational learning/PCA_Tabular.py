import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


X = np.load('../Data/X_train_tabular.npy')
y = np.load('../Data/y_train_tabular.npy')

# downSampling
sample_size = 10000
if X.shape[0] > sample_size:
    X, y = shuffle(X, y, random_state=42)
    X = X[:sample_size]
    y = y[:sample_size]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction to 3 dimensions
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# t-SNE for dimensionality reduction to 3 dimensions
tsne = TSNE(n_components=3, random_state=42, max_iter=300, perplexity=30, early_exaggeration=12)
X_tsne = tsne.fit_transform(X_scaled)

# label
y_single_label = y[:, 0]

# Check shapes for consistency
print(f'X_pca shape: {X_pca.shape}')
print(f'X_tsne shape: {X_tsne.shape}')
print(f'y_single_label shape: {y_single_label.shape}')

# Plot PCA results
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_single_label, cmap='viridis', alpha=0.5)
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('PCA Result (3D)')
plt.colorbar(scatter, label='Target Value')
plt.savefig('PCA_3D.png')
plt.show()

# Plot t-SNE results
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y_single_label, cmap='viridis', alpha=0.5)
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('t-SNE Result (3D)')
plt.colorbar(scatter, label='Target Value')
plt.savefig('tSNE_3D.png')
plt.show()
