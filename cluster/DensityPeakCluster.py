import math
import sys
import logging
import numpy
import numpy as np

from cluster.plot import plot_rho_delta

logger = logging.getLogger("dpc_cluster")

def load_paperdata(distance_f):
    distances = {}
    min_dis, max_dis = sys.float_info.max, 0.0
    max_id = 0

    for line in distance_f:
        x1, x2, d = line
        x1, x2 = int(x1), int(x2)
        max_id = max(max_id, x1, x2)
        dis = float(d)
        min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
        distances[(x1, x2)] = float(d)
        distances[(x2, x1)] = float(d)
    return distances, max_dis, min_dis, max_id



def compute_standard_deviation(max_id, dc, distances):
    loc_density, _ = local_density(max_id, distances, dc)
    loc_density = loc_density[1:]
    loc_density_normalized = (loc_density - np.min(loc_density)) / (np.max(loc_density) - np.min(loc_density))
    loc_density_mean = sum(loc_density_normalized) / len(loc_density_normalized)
    squared_differences = [(x - loc_density_mean) ** 2 for x in loc_density_normalized]
    variance = sum(squared_differences) / (len(loc_density_normalized) - 1)
    sigma = math.sqrt(variance)
    return sigma




def autoselect_dc(max_id, max_dis, min_dis, distances):
    '''
    :param max_id:
    :param max_dis:
    :param min_dis:
    :param distances:
    :return:
    '''
    max_sigma = -1
    for dc in np.arange(min_dis + 0.01, max_dis, 0.01):
        print(f"dc={dc}")
        loc_density, distance_matrix = local_density(max_id, distances, dc)
        loc_density = loc_density[1:]
        loc_density_normalized = (loc_density - np.min(loc_density)) / (
                np.max(loc_density) - np.min(loc_density))
        loc_density_mean = sum(loc_density_normalized) / len(loc_density_normalized)
        squared_differences = [(x - loc_density_mean) ** 2 for x in loc_density_normalized]
        variance = sum(squared_differences) / (len(loc_density_normalized) - 1)
        sigma = math.sqrt(variance)

        if sigma > max_sigma:
            max_sigma = sigma
            best_dc = dc

    return best_dc



def local_density(max_id, distances, dc, guass=True, cutoff=False):
    assert guass and not cutoff or not guass and cutoff
    logger.info("PROGRESS: compute local density")
    if guass:
        func = lambda dij, dc: np.exp(- (dij / dc) ** 2)
    else:
        func = lambda dij, dc: np.where(dij < dc, 1, 0)

    rho1 = np.zeros(max_id)
    distance_matrix = np.zeros((max_id, max_id))

    for (i, j), value in distances.items():
        distance_matrix[i - 1, j - 1] = value
        distance_matrix[j - 1, i - 1] = value
    for i in range(0, max_id):
        dij = np.hstack((distance_matrix[i, 0:i], distance_matrix[i, i + 1:]))
        rho_i = np.sum(func(dij, dc))
        rho1[i] = rho_i

    if max_id > 10:
        logger.info("PROGRESS: at index #%i" % max_id)

    rho1 = np.concatenate(([-1], rho1)).astype(np.float32)

    return rho1, distance_matrix


def min_distance(max_id, max_dis, distances, local_density, distance_matrix):
    logger.info("PROGRESS: compute min distance to nearest higher density neigh")
    sort_rho_idx = np.argsort(-local_density)
    delta1 = np.full(len(local_density), float(max_dis), dtype=np.float32)
    delta1[0] = 0
    nneigh1 = np.zeros(len(local_density), dtype=np.int32)

    delta1[sort_rho_idx[0]] = -1.
    for i in range(1, max_id):
        old_i = sort_rho_idx[i]
        old_j_values = sort_rho_idx[:i]
        distances_ij = distance_matrix[old_i - 1, old_j_values - 1]

        min_distance = np.min(distances_ij)
        min_distance_index = np.argmin(distances_ij)
        delta1[old_i] = min_distance
        nneigh1[old_i] = old_j_values[min_distance_index]

        if i % (max_id / 10) == 0:
            logger.info("PROGRESS: at index #%i" % (i))

    delta1[sort_rho_idx[0]] = np.max(delta1)

    return delta1, nneigh1


def calculate_cluster_num(gamma):
    gamma = numpy.matrix.tolist(gamma)
    gamma.sort(reverse=True)
    K = []

    for i in range(len(gamma) - 1):
        k = gamma[i] - gamma[i + 1]
        K.append(k)
    ksum = 0
    for i in range(len(K)):
        ksum = ksum + K[i]
    R = ksum / len(K)
    Result = 1
    for i in range(len(K)):
        if K[i] > R:
            Result = Result + 1

    return Result


def cluster_result(local_density, delta, nneigh):
    local_density = local_density[1:]
    delta = delta[1:]
    nneigh = nneigh[1:]
    nneigh = [int(x - 1) if x > 0 else 0 for x in nneigh]

    normal_den = (local_density - np.min(local_density)) / (np.max(local_density) - np.min(local_density))
    normal_dis = (delta - np.min(delta)) / (np.max(delta) - np.min(delta))
    gamma = normal_den * normal_dis

    plot_rho_delta(normal_den, normal_dis)
    sorted_indices = np.argsort(gamma)[::-1]
    #classNum = int(input("please input the number of cluster: "))

    # provide a method to determine the number of cluster
    classNum = calculate_cluster_num(gamma)

    densitySortArr = np.argsort(local_density)

    corePoxints, labels = extract_cluster(densitySortArr, nneigh, classNum, gamma)

    return labels


def extract_cluster(densitySortArr, closestNodeIdArr, classNum, gamma):
    n = densitySortArr.shape[0]
    labels = np.full((n,), -1)
    corePoints = np.argsort(-gamma)[: classNum]
    labels[corePoints] = range(len(corePoints))
    densitySortList = densitySortArr.tolist()
    densitySortList.reverse()
    for nodeId in densitySortList:
        if labels[nodeId] == -1:
            labels[int(nodeId)] = labels[int(closestNodeIdArr[int(nodeId)])]

    indices_of_minus_one = np.where(labels == -1)[0]
    while len(indices_of_minus_one) > 0:
        for nodeId in indices_of_minus_one:
            labels[int(nodeId)] = labels[int(closestNodeIdArr[int(nodeId)])]
        indices_of_minus_one = np.where(labels == -1)[0]

    return corePoints, labels


class DensityPeakCluster():

    def local_density(self, distance_f, dc=None, auto_select_dc=True):

        distances, max_dis, min_dis, max_id = load_paperdata(distance_f)
        best_dc = autoselect_dc(max_id, max_dis, min_dis, distances)
        loc_density, distance_matrix = local_density(max_id, distances, best_dc)

        return distances, max_dis, min_dis, max_id, loc_density, best_dc, distance_matrix

