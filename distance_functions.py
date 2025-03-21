import torch


class EuclideanDistance:
    def __init__(self, name="Euclidean Distance"):
        self.name = name

    def __call__(self, x):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
        return distances

    def get_settings(self):
        return {}

class CosineDistance:
    def __init__(self, name="Cosine Distance"):
        self.name = name

    def __call__(self, x):
        x_flat = x.view(x.size(0), -1)  # Flatten all but the first dimension

        # Normalize each row to unit length
        x_norm = x_flat / (torch.norm(x_flat, dim=1, keepdim=True) + 1e-8)  # Avoid division by zero

        # Compute cosine similarity
        sim_matrix = x_norm @ x_norm.T

        # Convert similarity to distance
        dist_matrix = 1 - sim_matrix
        dist_matrix.fill_diagonal_(0)

        return torch.abs(dist_matrix)

    def get_settings(self):
        return {}

class L1Distance:
    def __init__(self, name="L1 Distance"):
        self.name = name

    def __call__(self, x):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=2)
        return distances

    def get_settings(self):
        return {}
    
class LpDistance:
    def __init__(self, p, name="LP Distance"):
        self.name = name
        self.p = p

    def __call__(self, x):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=self.p)
        return distances
    
    @staticmethod
    def extract_lp_value(s):
        import re
        match = re.search(r'lp=(\d+)', s)
        if match:
            return int(match.group(1))  # Extract and convert to integer
        return None
    def get_settings(self):
        return {"p":self.p}
#not working DO NOT USE
class UMAP_Distance:
    def __init__(self,n_neighbors = 15,target_perplexity=1.0,name = "UMAP_similarity_based_distance"):
        self.n_neighbors = n_neighbors
        self.target_perplexity = target_perplexity
        self.name = name
    
    def __call__(self,x):
        return self.umap_inspired_distance_function(X = x)
    
    def find_sigma(self,knn_distances,rho,n,device):
        sigma = torch.ones(n, device=device) * 1.0
        for _ in range(10):  # Iterate to refine sigma
            p_ij = torch.exp(- (knn_distances - rho[:, None]) / sigma[:, None])
            sum_p = p_ij.sum(dim=1)
            sigma = sigma * (sum_p / self.target_perplexity).pow(0.5)
        return sigma
        
    def umap_inspired_distance_function(self,X):
        """
        Compute the symmetric UMAP transition probability matrix.
        
        Parameters:
            X (torch.Tensor): n x d tensor of data points.
            n_neighbors (int): Number of neighbors to consider.
        
        Returns:
            torch.Tensor: Symmetric transition probability matrix (n x n).
        """
        n = X.shape[0]

        # Compute pairwise Euclidean distances (n x n)
        dist = torch.cdist(X, X, p=2)

        # Get k-nearest neighbors (excluding self)
        knn_distances, knn_indices = torch.topk(dist, k=self.n_neighbors+1, largest=False)
        knn_distances, knn_indices = knn_distances[:, 1:], knn_indices[:, 1:]  # Remove self

        # Compute local connectivity rho_i (minimum nonzero distance per point)
        rho = knn_distances[:, 0]

        # Compute sigma_i via binary search to ensure fixed perplexity-like behavior
        

        sigma = self.find_sigma(knn_distances=knn_distances,n=n,rho=rho,device=X.device)

        # Compute UMAP transition probabilities
        P = torch.exp(- (knn_distances - rho[:, None]) / sigma[:, None])

        # Normalize each row so sum is 1
        P /= P.sum(dim=1, keepdim=True)

        # Build sparse adjacency matrix
        P_sparse = torch.zeros(n, n, device=X.device, dtype=P.dtype)
        P_sparse = P_sparse.scatter(1, knn_indices, P)
        # for i in range(n):
        #     P_sparse[i, knn_indices[i]] = P[i]

        # Symmetrize: P_sym = (P + P^T) - (P * P^T)
        P_sym = P_sparse + P_sparse.T - P_sparse * P_sparse.T

        return 1 - P_sym - torch.eye(n,device = P_sym.device)

