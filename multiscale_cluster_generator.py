import numpy as np


def calculate_distance_matrix_for_pointcloud(pointcloud = None):
    if pointcloud is None:
        pointcloud = create_point_cloud_branching_clusters()
        print("point cloud not specified generating one with default parameters")
    if isinstance(pointcloud,list):
        pointcloud = np.vstack(pointcloud)
    diffs = pointcloud[:, np.newaxis, :] - pointcloud[np.newaxis, :, :]
    sq_diffs = np.sum(diffs**2, axis=-1)
    pairwise_distances = np.sqrt(sq_diffs)
    return pairwise_distances


def create_point_cloud_branching_clusters(number_of_clusters_at_each_scale = (3,3,3), nb_of_points_per_smallest_cluster = 5,dimension_of_space = 2,magnitude_difference_between_scales = 0.25):
    number_of_clusters_at_each_scale = list(number_of_clusters_at_each_scale)
    number_of_clusters_at_each_scale.append(nb_of_points_per_smallest_cluster)
    center_of_clusters,cluster_labels = calculate_cluster_centers(number_of_clusters_at_each_scale=number_of_clusters_at_each_scale,dimension_of_space=dimension_of_space,magnitude_difference_between_scales=magnitude_difference_between_scales)
    return np.array(center_of_clusters[-1]),np.array(cluster_labels[-1])

def calculate_cluster_centers(number_of_clusters_at_each_scale,dimension_of_space,magnitude_difference_between_scales):
    centers = [[np.zeros((dimension_of_space))]]
    labels = [[0]]
    scale = 1
    for number_centers in number_of_clusters_at_each_scale:
        new_centers = []
        new_labels = []
        for label,center in enumerate(centers[-1]):
            x = generate_n_points_around_center_with_avg_radius(number_of_points=number_centers,center_coordinates=center,dimension_of_space=dimension_of_space,radius=scale)
            new_centers.extend(x)
            new_labels.extend([label]*len(x))
        centers.append(new_centers)
        labels.append(new_labels)
        scale = scale * magnitude_difference_between_scales
    return centers,labels


def generate_n_points_around_center_with_avg_radius(number_of_points,center_coordinates,dimension_of_space,radius):
    points = []
    for point_index in range(number_of_points):
        point = np.random.normal(size=dimension_of_space)
        point /= np.linalg.norm(point)
        rng_radius = ((0.9 + 0.1*np.random.rand())**(1/dimension_of_space)) * radius
        points.append(center_coordinates + rng_radius * point)
    return points


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    coordinates,labels = create_point_cloud_branching_clusters()
    # x_coords = [arr[0] for arr in coordinates]
    # y_coords = [arr[1] for arr in coordinates]

    # plt.figure(figsize=(6, 6))
    # plt.scatter(x_coords, y_coords, color='blue', label='Points')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Scatter Plot of [x, y] Points')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    cmap = plt.colormaps['tab20'].resampled(num_labels) 
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        coordinates[:, 0],  # X-coordinates
        coordinates[:, 1],  # Y-coordinates
        c=labels,           # Use labels to color points
        cmap=cmap,          # Dynamically adapt colormap
        s=50,               # Marker size
        edgecolor='k'       # Marker edge color
    )
    cbar = plt.colorbar(scatter, ticks=range(num_labels), boundaries=np.arange(-0.5, num_labels + 0.5, 1))
    cbar.set_label('Labels')
    cbar.set_ticks(range(num_labels))
    cbar.set_ticklabels(unique_labels)

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatterplot of Points with Dynamically Adjusted Colormap")
    plt.show()
