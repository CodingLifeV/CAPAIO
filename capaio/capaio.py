from collections import defaultdict
import numpy as np

import constants
from cluster.DensityPeakCluster import DensityPeakCluster, min_distance, cluster_result
from distance.distance import SqrtDistance
from distance.distancebuilder import DistanceBuilder
from capaio.sampling import _synthetic_inland_samples, _synthetic_borderline_samples, calculate_systhetic_num, _synthetic_trapped_samples


def calculate_nearest_neighbors(instances, distances, distance_matrix, KNN):
    nearest_neighbors1 = defaultdict(list)
    for index_to_select in range(1, len(instances) + 1):

        distances_to_other_instances = distance_matrix[index_to_select - 1]
        nearest_neighbor_indices = np.argsort(distances_to_other_instances)
        nearest_neighbor_indices = nearest_neighbor_indices[1:]
        m_nearest_neighbors = [(i + 1, distances_to_other_instances[i]) for i in
                               nearest_neighbor_indices[:KNN]]
        nearest_neighbors1[index_to_select] = m_nearest_neighbors

    return nearest_neighbors1

def _same_cluster_neighbors_count(labels, nearest_neighbors):
    same_cluster_neighbors_count = defaultdict(int)

    for instance in range(len(labels)+1):
        same_cluster_neighbors_count[instance] = 0

    for instance, neighbor_data in nearest_neighbors.items():
        instance_cluster = labels[instance - 1]
        for neighbor in neighbor_data:
            neighbor_instance = neighbor[0]
            neighbor_cluster = labels[neighbor_instance - 1]
            if instance_cluster == neighbor_cluster:
                same_cluster_neighbors_count[instance] += 1

    if same_cluster_neighbors_count:
        first_key = next(iter(same_cluster_neighbors_count))
        del same_cluster_neighbors_count[first_key]

    return same_cluster_neighbors_count

def _instance_characterize(nearest_neighbors, local_density_x):
    instance_categories = []
    for instance, neighbor_data in nearest_neighbors.items():
        instance_density = local_density_x[instance - 1]
        instance_category = 1
        if instance_density > constants.Density_threshold:
            instance_category = 0
        else:
            for neighbor in neighbor_data:
                neighbor_instance = neighbor[0]
                if local_density_x[neighbor_instance - 1] > constants.Density_threshold:
                    instance_category = 2
                    break
        instance_categories.append(instance_category)

    return instance_categories


class CAPAIO():

    def __init__(self, minority_class=None, ):
        self.minority_class = minority_class

    def inland_trapped_borderline(self, instances, labels, distances, distance_matrix, KNN):
        cluster_result_minority = []
        for i in range(0, len(instances)):
            cluster_result_minority.append((labels[i], instances[i]))

        # instance categories：0（'Inland'）1（'Trapped'） 2（'Borderline'）
        nearest_neighbors = calculate_nearest_neighbors(instances, distances, distance_matrix, KNN)
        same_cluster_neighbors_count = _same_cluster_neighbors_count(labels, nearest_neighbors)
        local_density_x = [value / KNN for value in same_cluster_neighbors_count.values()]

        instance_categories = _instance_characterize(nearest_neighbors, local_density_x)

        return local_density_x, instance_categories, nearest_neighbors


    def synthetic_new_samples(self, minority_instances, majority_instances, labels, local_density_x, nearest_neighbors, instance_categories):
        num_synthetic = calculate_systhetic_num(local_density_x, minority_instances, majority_instances, labels)

        synthetic_inland_samples = _synthetic_inland_samples(minority_instances, labels, num_synthetic, instance_categories)
        # 2. Generate synthetic samples for the borderline samples
        synthetic_borderline_samples = _synthetic_borderline_samples(minority_instances, majority_instances,
                                                                     num_synthetic, instance_categories, constants.k2)
        # 3. Generate synthetic samples for the trapped samples
        synthetic_trapped_samples, majority_instances = _synthetic_trapped_samples(minority_instances,
                                                                                    majority_instances, labels,
                                                                                    num_synthetic, nearest_neighbors, instance_categories, constants.k3)

        return synthetic_inland_samples, synthetic_borderline_samples, synthetic_trapped_samples, majority_instances

    # resampling
    def fit_resample(self, X, y, is_binary):
        classes = np.unique(y)
        if self.minority_class is None:
            sizes = [sum(y == c) for c in classes]
            minority_class = classes[np.argmin(sizes)]
            if (is_binary):
                majority_class = classes[np.argmax(sizes)]
        else:
            minority_class = self.minority_class

        constants.minority_class = minority_class
        constants.majority_class = majority_class

        minority_instances = X[y == minority_class].copy()
        majority_instances = X[y != minority_class].copy()

        builder = DistanceBuilder()
        dpcluster = DensityPeakCluster()

        minority_instances_distances = \
            builder.build_distance_file_for_cluster(SqrtDistance(),minority_instances)

        distances, max_dis, min_dis, max_id, loc_density, dc, distance_matrix = \
            dpcluster.local_density(minority_instances_distances)

        delta, nneigh = min_distance(max_id, max_dis, distances, loc_density, distance_matrix)

        labels = cluster_result(loc_density, delta, nneigh)

        local_density_x, instance_categories, nearest_neighbors = self.inland_trapped_borderline(minority_instances, labels, distances, distance_matrix, constants.Nearest_neighbors)

        synthetic_inland_samples, synthetic_borderline_samples, synthetic_trapped_samples, majority_instances =\
            self.synthetic_new_samples(minority_instances, majority_instances, labels, local_density_x,
                                  nearest_neighbors, instance_categories)

        x_result, y_result = self.final_result(minority_instances, majority_instances, majority_class, minority_class,
                                               synthetic_inland_samples, synthetic_borderline_samples, synthetic_trapped_samples)

        return x_result, y_result

    def final_result(self, minority_instances, majority_instances, majority_class, minority_class,
                     synthetic_inland_samples, synthetic_borderline_samples, synthetic_trapped_samples):
        '''
        :param minority_instances:
        :param majority_instances:
        :param majority_class:
        :param minority_class:
        :param synthetic_inland_samples:
        :param synthetic_borderline_samples:
        :param synthetic_trapped_samples:
        :return: x_result, y_result
        '''
        x_result = np.empty((0, minority_instances.shape[1]))
        y_result = []
        if len(majority_instances) != 0:
            x_result = np.concatenate((x_result, majority_instances), axis=0)
            for _ in range(len(majority_instances)):
                y_result.append(majority_class)
        if len(minority_instances) != 0:
            x_result = np.concatenate((x_result, minority_instances), axis=0)
            for _ in range(len(minority_instances)):
                y_result.append(minority_class)
        if len(synthetic_inland_samples) != 0:
            x_result = np.concatenate((x_result, synthetic_inland_samples), axis=0)
            for _ in range(len(synthetic_inland_samples)):
                y_result.append(minority_class)
        if len(synthetic_borderline_samples) != 0:
            x_result = np.concatenate((x_result, synthetic_borderline_samples), axis=0)
            for _ in range(len(synthetic_borderline_samples)):
                y_result.append(minority_class)
        if len(synthetic_trapped_samples) != 0:
            x_result = np.concatenate((x_result, synthetic_trapped_samples), axis=0)
            for _ in range(len(synthetic_trapped_samples)):
                y_result.append(minority_class)
        return x_result, y_result









