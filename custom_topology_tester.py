from .multiscale_cluster_generator import calculate_distance_matrix_for_pointcloud
from .multi_scale_connectivity_encoder import ConnectivityEncoderCalculator


if __name__ == "__main__":

    distance_mat = calculate_distance_matrix_for_pointcloud()
    topo_encoding = ConnectivityEncoderCalculator(distance_mat)
    topo_encoding.calculate_connectivity()
    x = topo_encoding.what_connected_these_two_points(4,7)
    x = topo_encoding.what_connected_this_point_to_this_set(4,[50,51])
    print(x)
    pass