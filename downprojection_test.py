import sys

from .connectivity_dp_experiment import DATASETS



sys.path.append("C:/Users\MiladBassil/Desktop/Master_Thesis/code\Master_Thesis_Code")
import numpy as np

np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import phate
from sklearn.datasets import fetch_openml, make_swiss_roll
from .multiscale_cluster_generator import create_point_cloud_branching_clusters
from .downprojection_tool import ConnectivityDP


DATASET = DATASETS[2]
NB_SAMPLES = 200


# Load MNIST dataset
def perform_mnist_test():
    if DATASET == DATASETS[0]:
        print("Loading MNIST dataset...")
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist.data, mnist.target.astype(int)
        indices = np.random.choice(X.shape[0], NB_SAMPLES, replace=False)
        X, y = X.to_numpy(), y.to_numpy()
        X, y = X[indices], y[indices]
        X = X / 255.0
    elif DATASET == DATASETS[1]:
        print("Generating Swiss Roll dataset...")
        X, y = make_swiss_roll(n_samples=NB_SAMPLES, random_state=42)
        y = np.floor(y).astype(int)
    elif DATASET == DATASETS[2]:
        from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
        print("Loading AML dataset...")
        data = np.load(r"data\SCEMILA\dinobloom_feature_data\dinbloomS_cell_level_label.npz")
        X = data["embedding"]
        y_string = data["labels"]
        indices = np.arange(1500)  # Create an array of indices
        np.random.shuffle(indices)  # Shuffle the indices
        x_shuffled = X[indices]  # Shuffle x using the shuffled indices
        y_shuffled = y_string[indices]  # Shuffle y using the same indices
        X = x_shuffled[:NB_SAMPLES]
        y_string = y_shuffled[:NB_SAMPLES]
        y = np.zeros(y_string.shape)
        indexer = SCEMILA_Indexer()

        for index in range(NB_SAMPLES):
            y[index] = indexer.convert_from_int_to_label_instance_level(y_string[index])
        y = y_string
            
    elif DATASET == DATASETS[3]:
        print("Generating Clusters dataset...")
        X, y = create_point_cloud_branching_clusters(dimension_of_space=10,nb_of_points_per_smallest_cluster=int(NB_SAMPLES/27))
    else:
        raise ValueError("Unsupported dataset")
    print("Calculating Connectivity DP...")
    connectivity_operator = ConnectivityDP(
        n_iter=1000,
        learning_rate=0.01,
        normalize_input=True,
        loss_calculation_timeout=10,
        optimizer_name="adan",
        importance_calculation_strat='min',
        augmentation_scheme={"name": "uniform", "p": 0.01},
        dev_settings={"labels": y, "create_vid": None, "++moor_method": None, "dataset_name": DATASET},
    )
    connectivity_embedding = connectivity_operator.fit_transform(X)
    print("Calculating PCA...")
    pca_embedding = PCA(n_components=2).fit_transform(X)

    print("Calculating t-SNE...")
    tsne_embedding = TSNE(n_components=2, init="random", random_state=42).fit_transform(X)

    print("Calculating UMAP...")
    umap_embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

    print("Calculating PHATE...")
    phate_operator = phate.PHATE(n_components=2, random_state=42)
    phate_embedding = phate_operator.fit_transform(X)

    # Plot the results
    def plot_embeddings(embeddings, titles, labels, figsize=(12, 8)):
        """Helper function to visualize embeddings."""
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"{DATASET} Dimensionality Reduction", fontsize=16)
        
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        cmap = plt.colormaps["tab20"].resampled(num_labels)
        
        # Create subplots
        for i, (embedding, title) in enumerate(zip(embeddings, titles)):
            plt.subplot(2, 3, i + 1)
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=5)
            plt.title(title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        
        # Add a single colorbar for all subplots
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()

    # Visualize
    plot_embeddings(
        embeddings=[
            pca_embedding,
            tsne_embedding,
            umap_embedding,
            phate_embedding,
            connectivity_embedding,
        ],
        titles=["PCA", "t-SNE", "UMAP", "PHATE", "Connectivity"],
        labels=y,
    )

def test_connectivity_encoding():
    from models.topology_models.custom_topo_tools.topo_encoder import ConnectivityEncoderCalculator
    from models.topology_models.topo_tools.topology import PersistentHomologyCalculation
    from scipy.spatial.distance import cdist
    X, y = create_point_cloud_branching_clusters()
    distance_matrix = cdist(X, X, metric='euclidean')
    mine_c = ConnectivityEncoderCalculator(distance_matrix)
    mine_c.calculate_connectivity()
    mine  = mine_c.persistence_pairs
    for index in range(len(mine)):
        print(f"{mine_c.component_total_importance_score[index]:.4g}")
        #print(f"scale,component_size_imp,persistence_imp,mult_score: {mine_c.scales[index]:.4g},{mine_c.pers_pair_importance_score[index]:.4g},{mine_c.persistence_of_components_scaled[index]:.4g},{mine_c.pers_pair_importance_score[index]*mine_c.persistence_of_components_scaled[index]:.4g}")
    theirs = PersistentHomologyCalculation()(distance_matrix)[0] #since it assumes 1 degree topo implemented
    theirs = [tuple(pair) for pair in theirs]
    # print(f"mine: {mine}")
    # print(f"theirs: {theirs}")
    # print(mine==theirs)
if __name__ == "__main__":
    perform_mnist_test()
    #test_connectivity_encoding()