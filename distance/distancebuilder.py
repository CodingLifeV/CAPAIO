

class DistanceBuilder(object):
    """
    Build distance file for cluster
    """

    def __init__(self):
        self.vectors = []

    def load_points(self, minority_instances):
        self.vectors = minority_instances

    def build_distance_file_for_cluster(self, distance_obj,minority_instances):

        distances = []
        self.vectors = minority_instances

        for i in range(len(self.vectors)):
            for j in range(i, len(self.vectors)):
                distance = distance_obj.distance(self.vectors[i], self.vectors[j])
                distances.append((i + 1, j + 1, distance))


        return distances
