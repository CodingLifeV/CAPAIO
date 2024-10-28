import numpy as np
import random
import math
from collections import defaultdict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

import constants
from distance.distance import SqrtDistance


def fix_distribution(dist):
    """
    Fixing a distribution.

    Args:
        dist (iterable): the distribution

    Returns:
        np.array: the fixed distribution
    """
    dist = np.nan_to_num(dist, nan=0.0)
    if np.sum(dist) == 0:
        return np.repeat(1.0, len(dist))
    return dist


def adaptive_sub_cluster_sizing(labels, minority_instances, majority_instances
                                ,majority_class, minority_class):
    '''
    :param labels:
    :param minority_instances:
    :param majority_instances:
    :return:
    '''
    eps = []

    min_clusters, min_clusters_counts = np.unique(labels, return_counts=True)

    for clus in min_clusters:
        if min_clusters_counts[clus] > 1:
            preds = []
            X_c = minority_instances[labels == clus]

            kfold = KFold(min([min_clusters_counts[clus], constants.K_FLOD, len(X_c)]),
                          random_state=42,
                          shuffle=True)
            for train, test in kfold.split(X_c):
                X_train = np.vstack([majority_instances, X_c[train]])
                y_train = np.hstack([np.repeat(majority_class, len(majority_instances)),
                                 np.repeat(minority_class, len(X_c[train]))])
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda.fit(X_train, y_train)

                preds.append(lda.predict(X_c[test]))
            preds = np.hstack(preds)

            # extracting error rate
            eps.append(np.sum(preds == majority_class) / len(preds))
        else:
            eps.append(1.0)

    eps = fix_distribution(eps)

    return eps / np.sum(eps)




def calculate_systhetic_num(local_density_x, minority_instances, majority_instances, labels):
    '''
    :param local_density_x:
    :param minority_instances:
    :param majority_instances:
    :param labels:
    :return:
    '''

    sub_cluster_sizes = adaptive_sub_cluster_sizing(labels, minority_instances, majority_instances, constants.majority_class,
                                                    constants.minority_class)

    # k majority nearest neighbors
    nearest_neighbors_majority = _nearest_neighbors_majority(minority_instances, majority_instances, constants.Nearest_neighbors_maj)
    for index, values in nearest_neighbors_majority.items():
        for i, (neighbor_index, distance) in enumerate(values):
            if distance == 0:
                nearest_neighbors_majority[index][i] = (neighbor_index, 1)
            else:
                nearest_neighbors_majority[index][i] = (neighbor_index, 1 / (distance / len(minority_instances[0])))

    TH = []
    cluster_num = np.unique(labels)
    counts = np.bincount(labels)
    for x in cluster_num:
        distances = 0
        for j in range(len(minority_instances)):
            if (labels[j] == x):
                distances = distances + nearest_neighbors_majority[j][0][1]
        TH.append(distances / counts[np.where(cluster_num == x)[0]])

    # compute closeness factor
    closeness = [0] * len(minority_instances)
    for idx in range(len(minority_instances)):
        closeness[idx] = nearest_neighbors_majority[idx][0][1]
        if (closeness[idx] > TH[labels[idx]]):
            closeness[idx] = TH[labels[idx]][0]

    closeness = np.array(closeness)
    for idx in cluster_num:
        closeness_sum = sum(closeness[labels == idx])
        closeness[labels == idx] = closeness[labels == idx] / closeness_sum

    num_synthetic1 = []
    sub_cluster_synthetic = sub_cluster_sizes * (len(majority_instances) - len(minority_instances))
    for i in range(0, len(minority_instances)):
        pos = np.where(cluster_num == labels[i])[0]
        num_synthetic1.append(
            math.ceil(
                constants.beta * closeness[i] * sub_cluster_synthetic[pos]
            )
        )

    return num_synthetic1

'''
    inland 0
'''
def _synthetic_inland_samples(minority_instances, labels, num_synthetic, instance_categories):
    synthetic_inland_samples = []
    for i in range(len(minority_instances)):
        if instance_categories[i] == 0:
            for _ in range(num_synthetic[i]):
                gamma = random.uniform(0, 1)

                candidate_indices = [j for j in range(len(minority_instances)) if labels[j] == labels[i] and j != i]

                candidate_index = random.choice(candidate_indices)
                candidate_sample = minority_instances[candidate_index]

                synthetic_sample = minority_instances[i] + gamma * (candidate_sample - minority_instances[i])
                synthetic_inland_samples.append(synthetic_sample)
    return synthetic_inland_samples


'''
    borderline 2
'''
def _synthetic_borderline_samples(minority_instances, majority_instances, num_synthetic, instance_categories, KNN):
    synthetic_borderline_samples = []
    for index, minority_instance in enumerate(minority_instances):
        if instance_categories[index] == 2:
            nearest_neighbors = []
            for i, majority_instance in enumerate(majority_instances):
                dist = SqrtDistance().distance(minority_instance, majority_instance)
                nearest_neighbors.append((i, dist))

            nearest_neighbors.sort(key=lambda x: x[1])
            m_nearest_neighbors = nearest_neighbors[:KNN]

            for _ in range(num_synthetic[index]):
                random_index = random.choice(m_nearest_neighbors)
                candidate_index = random_index[0]
                candidate_sample = majority_instances[candidate_index]

                synthetic_sample = minority_instance + random.uniform(0, 1) * (candidate_sample - minority_instance)
                synthetic_borderline_samples.append(synthetic_sample)
    return synthetic_borderline_samples


def _synthetic_trapped_samples(minority_instances, majority_instances, labels, num_synthetic, nearest_neighbors, instance_categories,KNN):
    synthetic_trapped_samples = []
    translation_majority_instances = []
    for index, minority_instance in enumerate(minority_instances):
        if instance_categories[index] == 1:
            nearest_neighbor = []
            for i, majority_instance in enumerate(majority_instances):
                dist = SqrtDistance().distance(minority_instance, majority_instance)
                nearest_neighbor.append((i, dist))

            nearest_neighbor.sort(key=lambda x: x[1])
            m_nearest_neighbors = nearest_neighbor[:KNN]

            if m_nearest_neighbors[-1][1] != 0:
                translation_majority_instances = transition_majority(majority_instances, minority_instances, index, m_nearest_neighbors)

                for _ in range(num_synthetic[index]):
                    random_index = random.choice(m_nearest_neighbors)
                    candidate_index = random_index[0]
                    candidate_sample = translation_majority_instances[candidate_index]

                    synthetic_sample = minority_instance + random.uniform(0, 1) * (candidate_sample - minority_instance)
                    synthetic_trapped_samples.append(synthetic_sample)

    if(len(translation_majority_instances) == 0):
        translation_majority_instances = majority_instances.copy()
    return synthetic_trapped_samples, translation_majority_instances

def transition_majority(majority_instances, minority_instances, index, m_nearest_neighbors):
    translation_majority_instances = majority_instances.copy()
    last_instance = m_nearest_neighbors[-1]
    gamma = []
    for neighbor in m_nearest_neighbors:
        idx, distance = neighbor
        gamma = distance / last_instance[1]
        translation_majority_instances[idx] = (majority_instances[idx] - minority_instances[index]) / gamma + minority_instances[index]
    return translation_majority_instances


def _trapped_neighbors(labels, nearest_neighbors):
    trapped_samples = {i for i, label in enumerate(labels) if label == 1}
    trapped_neighbors = {}

    for trapped_sample in trapped_samples:
        neighbors = nearest_neighbors[trapped_sample + 1]
        trapped_neighbors_set = set()
        for neighbor, distance in neighbors:
            neighbor_index = neighbor[1] - 1

            if(labels[trapped_sample] == labels[neighbor_index]):
                trapped_neighbors_set.add(neighbor_index)

        trapped_neighbors[trapped_sample] = trapped_neighbors_set

    return trapped_neighbors


def _nearest_neighbors_majority(minority_instances, majority_instances, KNN):

    nearest_neighbors_majority = defaultdict(list)

    for i, minority_instance in enumerate(minority_instances):
        nearest_indices_and_distances = []
        for j, majority_instance in enumerate(majority_instances):
            distance = SqrtDistance().distance(minority_instance, majority_instance)
            nearest_indices_and_distances.append((j, distance))

        nearest_indices_and_distances.sort(key=lambda x: x[1])
        k_nearest_neighbors = nearest_indices_and_distances[:KNN]

        nearest_neighbors_majority[i] = k_nearest_neighbors

    return nearest_neighbors_majority

def _majority_neighbors_trapped(trapped_neighbors, nearest_neighbors_majority, minority_instances):

    majority_neighbor_all = set()
    majority_neighbor_i = {}
    for i, trapped_neighbors_index in trapped_neighbors.items():
        majority_neighbor = set()
        for index in trapped_neighbors_index:
            for neighbor in nearest_neighbors_majority[index]:
                neighbor_index = neighbor[0]
                if neighbor_index not in majority_neighbor:
                    majority_neighbor.add(neighbor_index)

        majority_neighbor_i[i] = majority_neighbor
        majority_neighbor_all = majority_neighbor_all.union(majority_neighbor)

    return majority_neighbor_i, majority_neighbor_all



