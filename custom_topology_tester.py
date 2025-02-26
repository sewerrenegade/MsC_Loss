from .multiscale_cluster_generator import calculate_distance_matrix_for_pointcloud
from .topo_encoder import ConnectivityEncoderCalculator


if __name__ == "__main__":

    distance_mat = calculate_distance_matrix_for_pointcloud()
    topo_encoding = ConnectivityEncoderCalculator(distance_mat)
    topo_encoding.calculate_connectivity()
    x = topo_encoding.what_connected_these_two_points(4,7)
    pass