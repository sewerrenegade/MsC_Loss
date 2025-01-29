from matplotlib import gridspec, pyplot as plt
from distance_functions.distance_function_metrics.static_distance_matrix_metrics import StaticDistanceMatrixMetricCalculator
import numpy as np
np.random.seed(42)
from datasets.SCEMILA.SEMILA_indexer import SCEMILA_Indexer
from models.topology_models.custom_topo_tools.downprojection_tool import ConnectivityDP
from scipy.spatial.distance import cdist
from models.topology_models.custom_topo_tools.multiscale_cluster_generator import (
    create_point_cloud_branching_clusters,
)
from sklearn.datasets import fetch_openml, make_swiss_roll
DATASETS = ["MNIST", "SWISS_ROLL", "DinoBloom", "CLUSTERS"]

class ConnectivityHyperParamExperiment:
    def __init__(
        self,
        dataset_name="CLUSTERS",
        optimizer_name="sgd",
        LR=1.0,
        normalize_input=True,
        importance_weighting_strat="none",
        augmentation_strength=0.01,
        size_of_data=200,
        weight_decay=0.0,
    ):
        self.dataset_name = dataset_name
        self.optimizer_name = optimizer_name
        self.LR = LR
        self.normalize_input = normalize_input
        self.importance_weighting_strat = importance_weighting_strat
        self.augmentation_strength = augmentation_strength
        self.size_of_data = size_of_data
        self.weight_decay = weight_decay
        self.X, self.y = self.get_dataset()
        self.connectivity_dp = ConnectivityDP(
            n_components=2,
            n_iter=10,
            learning_rate=self.LR,
            optimizer_name=self.optimizer_name,
            normalize_input=self.normalize_input,
            weight_decay=self.weight_decay,
            loss_calculation_timeout=10,
            augmentation_scheme={"name": "uniform", "p": self.augmentation_strength},
            importance_calculation_strat=self.importance_weighting_strat,
        )
        
    def produce_2d_plot_of_embeddings(self, X, labels):
        """
        Creates a scatter plot with color and marker coding for different classes.

        Parameters:
        X (numpy.ndarray): A 2D array of shape (n_samples, 2), containing the points to plot.
        labels (list or numpy.ndarray): Class labels corresponding to the rows of `X`.
        """
        if X.shape[1] != 2:
            raise ValueError("Input data X must have exactly 2 columns for a 2D scatter plot.")

        # Create the figure and scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.set_title("Scatter Plot with Class Labels")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        # Define marker styles and colors
        marker_styles = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'X', 'h', 'H', '8', '|', '_', '.', ',']
        colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors  # Combined color maps

        # Ensure unique combinations of colors and markers
        num_classes = len(set(labels))
        unique_labels = sorted(set(labels))
        if num_classes > len(colors) * len(marker_styles):
            raise ValueError("Not enough combinations of colors and markers for all classes!")

        # Plot each class with unique color and marker
        for idx, label in enumerate(unique_labels):
            color = colors[idx % len(colors)]
            marker = marker_styles[idx // len(colors) % len(marker_styles)]
            subset = X[np.array(labels) == label]
            ax.scatter(
                subset[:, 0],
                subset[:, 1],
                color=color,
                marker=marker,
                label=f"{label}",
                s=50,
                alpha=0.8,
                edgecolors="k",  # Optional edge color for better visibility
            )

        # Add legend outside the plot
        ax.legend(
            title="Classes",
            loc='upper left',
            bbox_to_anchor=(1.05, 1),  # Adjusts the legend position outside the plot
            borderaxespad=0
        )

        # Show the plot
        plt.tight_layout()
        return fig


    def run_experiment(self):
        connectivity_embedding = self.connectivity_dp.fit_transform(self.X)
        distance_matrix =  cdist(connectivity_embedding, connectivity_embedding)
        metrics = StaticDistanceMatrixMetricCalculator.calculate_distance_matrix_metrics(distance_matrix,self.y)
        metric_of_interest = {"intra_to_inter_class_distance_overall_ratio":metrics["intra_to_inter_class_distance_overall_ratio"],
                              "silhouette_score":metrics["silhouette_score"],
                              "loocv_knn_acc":metrics["loocv_knn_acc"],
                              "best_epoc": self.connectivity_dp.opt_epoch,
                              "best_loss": self.connectivity_dp.opt_loss
                              }
        try: 
            fig = self.produce_2d_plot_of_embeddings(connectivity_embedding,self.y)
        except Exception as e:
            fig = None
            print(f"Failed to produce the scatter plot visualization; {e}")
            
        return metric_of_interest, fig
    def get_dataset(self):
        if self.dataset_name == DATASETS[0]:
            print("Loading MNIST dataset...")
            mnist = fetch_openml("mnist_784", version=1)
            X, y = mnist.data, mnist.target.astype(int)
            indices = np.random.choice(X.shape[0], self.size_of_data, replace=False)
            X, y = X.to_numpy(), y.to_numpy()
            X, y = X[indices], y[indices]
            X = X / 255.0
        elif self.dataset_name == DATASETS[1]:
            print("Generating Swiss Roll dataset...")
            X, y = make_swiss_roll(n_samples=self.size_of_data, random_state=42)
        elif self.dataset_name == DATASETS[2]:
            print("Loading AML dataset...")
            data = np.load(r"data/SCEMILA/dinbloomS_labeled1.npz")
            X = data["embedding"]
            y_string = data["labels"]
            indices = np.arange(1500)
            np.random.shuffle(indices)
            x_shuffled = X[indices]
            y_shuffled = y_string[indices]

            # Extract the first 200 data points and their labels
            X = x_shuffled[: self.size_of_data]
            y_string = y_shuffled[: self.size_of_data]
            y = np.zeros(y_string.shape)
            indexer = SCEMILA_Indexer()

            for index in range(self.size_of_data):
                y[index] = indexer.convert_from_int_to_label_instance_level(
                    y_string[index]
                )
            y = y_string

        elif self.dataset_name == DATASETS[3]:
            print("Generating Clusters dataset...")
            X, y = create_point_cloud_branching_clusters(
                dimension_of_space=10,
                nb_of_points_per_smallest_cluster=int(self.size_of_data / 27),
            )
        else:
            raise ValueError("Unsupported dataset")
        return X, y
