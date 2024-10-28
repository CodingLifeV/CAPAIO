
import numpy as np

from math import sqrt
from abc import abstractmethod, ABCMeta

import numpy.linalg as linalg

from distance.WrongVecError import WrongVecError


class Distance():
    """
        abstract class, represent distance of two vector

        Attributes:
        """

    __metaclass__ = ABCMeta

    @abstractmethod
    def distance(self, vec1, vec2):
        """
        Compute distance of two vector(one line numpy array)
        if you use scipy to store the sparse matrix, please use s.getrow(line_num).toarray() build the one dimensional array

        Args:
            vec1: the first line vector, an instance of array
            vec2: the second line vector, an instance of array

        Returns:
            the computed distance

        Raises:
            TypeError: if vec1 or vec2 is not numpy.ndarray and one line array
        """
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            raise TypeError("type of vec1 or vec2 is not numpy.ndarray")
        if vec1.ndim is not 1 or vec2.ndim is not 1:
            raise WrongVecError("vec1 or vec2 is not one line array")
        if vec1.size != vec2.size:
            raise WrongVecError("vec1 or vec2 is not same size")
        pass

class SqrtDistance(Distance):
    """
    Square distance

    a sub class of Distance
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by square distance
        """
        super(SqrtDistance, self).distance(vec1, vec2)  # super method
        vec = vec1 - vec2
        return sqrt(sum([pow(item, 2) for item in vec]))

class ConsineDistance(Distance):
    """
    consine distance
    a sub class of Distance
    """

    def distance(self, vec1, vec2):
        """
        Compute distance of two vector by consine distance
        """
        super(ConsineDistance, self).distance(vec1, vec2)  # super method
        num = np.dot(vec1, vec2)
        denom = linalg.norm(vec1) * linalg.norm(vec2)
        if num == 0:
            return 1
        return - num / denom
