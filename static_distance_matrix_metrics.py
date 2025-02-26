import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score,cross_val_predict
import umap
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import random
from PIL import Image
from collections import defaultdict
import io

class StaticDistanceMatrixMetricCalculator:
    def __init__(self) -> None:
        self.name =  "distance_matrix_metrics"
        self.output_metrics_format = {
            "intra_to_inter_class_distance_overall_ratio": "scalar",
            "intra_to_inter_class_distance_overall_per_class_ratio":"dict-scalar",
            "intra_inter_class_distance_matrix_mean":"matrix",
            "intra_inter_class_distance_matrix_std":"matrix",
            "triplet_loss": "scalar",
            "silhouette_score": "scalar",
            "loocv_knn_acc": "scalar",
            "loocv_knn_acc_std": "scalar",
            "loocv_knn_report": "report",
            "loocv_confusion_matrix": "matrix",
            "knn_acc": "scalar",
            "knn_acc_std": "scalar",
            "knn_report": "report",
            "knn_confusion_matrix": "matrix"
        }
        
    @staticmethod
    def calculate_distance_matrix_metrics(distance_matrix,labels):
        overall_ratio, class_wise_ratios = StaticDistanceMatrixMetricCalculator.average_inter_intra_class_distance_ratio(distance_matrix, labels)
        intra_inter_class_distance_matrix_mean, intra_inter_class_distance_matrix_std = StaticDistanceMatrixMetricCalculator.average_inter_intra_class_distance_matrix(distance_matrix,labels)
        triplet_loss = StaticDistanceMatrixMetricCalculator.evaluate_triplet_loss(distance_matrix, labels)
        silhouette_score = StaticDistanceMatrixMetricCalculator.compute_silhouette_score_from_distance_matrix(distance_matrix, labels)
        loocv_knn_acc, loocv_knn_acc_std, loocv_knn_report, loocv_confusion_matrix = StaticDistanceMatrixMetricCalculator.knn_loocv_accuracy(distance_matrix, labels)
        knn_acc, knn_acc_std, knn_report, knn_confusion_matrix = StaticDistanceMatrixMetricCalculator.evaluate_knn_classifier_from_distance_matrix(distance_matrix, labels)

        metrics_dict = {
            "intra_to_inter_class_distance_overall_ratio": overall_ratio,
            "intra_to_inter_class_distance_overall_per_class_ratio":class_wise_ratios,
            "intra_inter_class_distance_matrix_mean":intra_inter_class_distance_matrix_mean,
            "intra_inter_class_distance_matrix_std":intra_inter_class_distance_matrix_std,
            "triplet_loss": triplet_loss,
            "silhouette_score": silhouette_score,
            "loocv_knn_acc": loocv_knn_acc,
            "loocv_knn_acc_std": loocv_knn_acc_std,
            "loocv_knn_report": loocv_knn_report,
            "loocv_confusion_matrix": loocv_confusion_matrix,
            "knn_acc": knn_acc,
            "knn_acc_std": knn_acc_std,
            "knn_report": knn_report,
            "knn_confusion_matrix": knn_confusion_matrix
        }

        return metrics_dict
    @staticmethod
    def calculate_metric():
        return StaticDistanceMatrixMetricCalculator.calculate_distance_matrix_metrics()
  

    @staticmethod
    def triplet_loss(anchor_idx, positive_idx, negative_idx, distance_matrix, margin=1.0):
        pos_dist = distance_matrix[anchor_idx, positive_idx]
        neg_dist = distance_matrix[anchor_idx, negative_idx]
        return max(0, pos_dist - neg_dist + margin)
    @staticmethod
    def evaluate_triplet_loss(distance_matrix, labels, margin=1.0):
        unique_labels = np.unique(labels)
        triplet_losses = []
        for label in unique_labels:
            class_indices = np.where(labels == label)[0]
            if len(class_indices) < 2:
                continue
            non_class_indices = np.where(labels != label)[0]
            for anchor_idx in class_indices:
                positive_idx = np.random.choice(class_indices[class_indices != anchor_idx])
                negative_idx = np.random.choice(non_class_indices)
                triplet_losses.append(StaticDistanceMatrixMetricCalculator.triplet_loss(anchor_idx, positive_idx, negative_idx,distance_matrix, margin = margin))
        return np.mean(triplet_losses)
    @staticmethod
    def compute_silhouette_score_from_distance_matrix(distance_matrix, labels):
        return silhouette_score(distance_matrix, labels, metric='precomputed')
    @staticmethod
    def  visualize_embeddings_from_distance_matrix(distance_matrix, labels, method='umap'):
        if method == 'tsne':
            embedding = TSNE(metric='precomputed').fit_transform(distance_matrix)
        elif method == 'umap':
            embedding = umap.UMAP(metric='precomputed').fit_transform(distance_matrix)
        else:
            raise ValueError("Unsupported method. Use 'tsne' or 'umap'.")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=100)
        plt.colorbar()
        
        # Convert plot to PIL image
        plt.tight_layout()
        fig_buf = io.BytesIO()
        plt.savefig(fig_buf, format='png')
        fig_buf.seek(0)
        plt.close()
        
        # Convert buffer to PIL image
        pil_image = Image.open(fig_buf)
        return pil_image


    @staticmethod
    def knn_loocv_accuracy(distance_matrix, labels, k=3):
        loo = LeaveOneOut()
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        
        # Perform LOOCV and collect predictions
        predictions = cross_val_predict(knn, distance_matrix, labels, cv=loo)
        
        # Calculate accuracy and std
        accuracy = np.mean(predictions == labels)
        std = np.std(predictions == labels)
        
        # Print classification report
        report = classification_report(labels, predictions,output_dict= True)
        conf_matrix = confusion_matrix(labels, predictions)
        
        return accuracy, std, report, conf_matrix
    @staticmethod
    def evaluate_knn_classifier_from_distance_matrix(distance_matrix, labels, k=3):
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
        knn.fit(distance_matrix, labels)
        predictions = knn.predict(distance_matrix)
        report = classification_report(labels, predictions,output_dict= True)
        accuracy = np.mean(predictions == labels)
        accuracy_std = np.std(predictions == labels)
        conf_matrix = confusion_matrix(labels, predictions)
        return accuracy,accuracy_std,report,conf_matrix
        
    @staticmethod
    def get_shuffled_list_data_from_class_indexed(data):
        list_data = []
        labels = []
        for key,value in data.items():
            list_data.extend(value)
            labels.extend([key]*len(value))
        combined = list(zip(list_data, labels))
        random.shuffle(combined)
        list_data_shuffled, labels_shuffled = zip(*combined)
        list_data_shuffled = list(list_data_shuffled)
        labels_shuffled = list(labels_shuffled)
        return list_data_shuffled,labels_shuffled
        
    @staticmethod
    def average_inter_intra_class_distance_ratio(distance_matrix, labels):
        n = len(labels)
        classes = np.unique(labels)
        class_wise_ratios = {}
        
        interclass_distances = []
        intraclass_distances = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j]:
                    interclass_distances.append(distance_matrix[i, j])
                else:
                    intraclass_distances[labels[i]].append(distance_matrix[i, j])
        
        if len(interclass_distances) == 0:
            overall_ratio = float('inf')  # No interclass distances available
        else:
            avg_interclass_distance = np.mean(interclass_distances)
            avg_intraclass_distance = np.mean([d for distances in intraclass_distances.values() for d in distances])
            overall_ratio = avg_intraclass_distance / avg_interclass_distance
        
        for cls in classes:
            if len(intraclass_distances[cls]) == 0:
                class_wise_ratios[cls] = float('inf')  # No intraclass distances for this class
            else:
                avg_intraclass_distance_cls = np.mean(intraclass_distances[cls])
                if len(interclass_distances) == 0:
                    class_wise_ratios[cls] = float('inf')
                else:
                    class_wise_ratios[cls] = avg_intraclass_distance_cls / avg_interclass_distance
        
        return overall_ratio, class_wise_ratios    
    @staticmethod
    def average_inter_intra_class_distance_matrix(distance_matrix, labels):
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        distance_matrix_mean = np.zeros((n_classes, n_classes))
        distance_matrix_std = np.zeros((n_classes, n_classes))
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        
        for i in range(n_classes):
            class_i = unique_labels[i]
            
            for j in range(n_classes):
                class_j = unique_labels[j]
                
                if i == j:  # Intraclass distance
                    intraclass_distances = []
                    indices_i = np.where(labels == class_i)[0]
                    
                    for k in range(len(indices_i)):
                        for l in range(k + 1, len(indices_i)):
                            intraclass_distances.append(distance_matrix[indices_i[k], indices_i[l]])
                    
                    if len(intraclass_distances) > 0:
                        avg_intraclass_distance = np.mean(intraclass_distances)
                        std_intraclass_distance = np.std(intraclass_distances)
                    else:
                        avg_intraclass_distance = 0
                        std_intraclass_distance = 0
                    
                    distance_matrix_mean[i, j] = avg_intraclass_distance
                    distance_matrix_std[i, j] = std_intraclass_distance
                
                else:  # Interclass distance
                    interclass_distances = []
                    indices_i = np.where(labels == class_i)[0]
                    indices_j = np.where(labels == class_j)[0]
                    
                    for k in indices_i:
                        for l in indices_j:
                            interclass_distances.append(distance_matrix[k, l])
                    
                    if len(interclass_distances) > 0:
                        avg_interclass_distance = np.mean(interclass_distances)
                        std_interclass_distance = np.std(interclass_distances)
                    else:
                        avg_interclass_distance = float('inf')
                        std_interclass_distance = 0
                    
                    distance_matrix_mean[i, j] = avg_interclass_distance
                    distance_matrix_std[i, j] = std_interclass_distance
        
        return distance_matrix_mean, distance_matrix_std

