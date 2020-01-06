from clustering.k_means.KMeans.AbstractKMeans import AbstractKMeans

import sys
import numpy as np
from copy import copy
from decimal import Decimal
import matplotlib.pyplot as plot


class MyKMeans(AbstractKMeans):
    def __init__(self, number_of_clusters, max_iterations=600, distance_type="euclidean"):
        super().__init__(number_of_clusters, max_iterations)
        self.__centroids = None
        self.__distance_type = distance_type

    @staticmethod
    def euclidean_distance(a, b):
        dist = 0
        for i in range(0, len(a)):
            dist += (a[i] - b[i]) ** 2
        return np.math.sqrt(dist)

    @staticmethod
    def manhattan_distance(a, b):
        dist = 0
        for i in range(0, len(a)):
            dist += abs(a[i] - b[i])
        return dist

    @staticmethod
    def p_root(value, root):
        return round(Decimal(value) ** Decimal(1 / float(root)), 3)

    @staticmethod
    def minkowski_distance(x, y, p_value=3):
        return MyKMeans.p_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)

    def distance(self, a, b):
        if self.__distance_type == "euclidean":
            return MyKMeans.euclidean_distance(a, b)
        elif self.__distance_type == "manhattan":
            return MyKMeans.euclidean_distance(a, b)
        elif self.__distance_type == "minkowski":
            return MyKMeans.minkowski_distance(a, b)
        else:
            return MyKMeans.euclidean_distance(a, b)

    def __initialize_centroids_randomly(self, dimensionality):
        # initialize the centroids with some random values between min and max for each dimension
        self.__centroids = []
        min_value = np.min(self._rows_pca)
        max_value = np.max(self._rows_pca)
        for i in range(0, self._number_of_clusters):
            centroid = []
            for j in range(0, dimensionality):
                centroid += [np.random.randint(min_value, max_value)]
            self.__centroids += [centroid]

    def __initialize_centroids_k_means_plus_plus(self):
        # initialize the centroid list
        # add a randomly selected data point to the list as the first centroid
        self.__centroids = []
        self.__centroids += [copy(self._rows_pca[np.random.randint(0, len(self._rows_pca) - 1)])]

        # compute remaining k - 1 centroids
        for k in range(self._number_of_clusters - 1):
            # initialize a list to store distances of data points from nearest centroid
            distances = []
            for i in range(0, len(self._rows_pca)):
                point = self._rows_pca[i]
                current_distance = sys.maxsize

                # compute distance of 'point' from each of the previously
                # selected centroids and store the minimum distance
                for j in range(0, len(self.__centroids)):
                    temp_distance = self.distance(point, self.__centroids[j])
                    current_distance = min(current_distance, temp_distance)
                distances.append(current_distance)

            # select data point with maximum distance as our next centroid
            distances = np.array(distances)
            next_centroid = copy(self._rows_pca[np.argmax(distances)])
            self.__centroids += [next_centroid]

    def calculate_clusters(self):
        # determine the initial dimensionality
        dimensionality = len(self._rows_pca[0])

        # initialize the centroids
        # self.__initialize_centroids_randomly(dimensionality)
        self.__initialize_centroids_k_means_plus_plus()

        # initially all the variables belong to the first cluster
        self._predicted_values = []
        distance_to_cluster = []
        max_distance = sys.maxsize
        for i in range(0, len(self._rows_pca)):
            self._predicted_values += [0]
            distance_to_cluster += [max_distance]

        # initialize a vector that counts how many points belong to each centroid
        point_belonging_count = []
        for i in range(0, self._number_of_clusters):
            point_belonging_count += [0]

        # initialize a vector for the sums of coordinates for the points belonging to each centroid
        point_belonging_sum = []
        for i in range(0, self._number_of_clusters):
            point_belonging_sum += [[]]
            for j in range(0, dimensionality):
                point_belonging_sum[i] += [0]

        # determine the nearest centroid for each point in the self._rows_pca array
        iteration_count = 0
        changed_value = True
        while iteration_count < self._max_iterations and changed_value:
            # initially, we haven't changed any point's closest centroid
            changed_value = False

            # initially, no point belongs to any cluster
            for i in range(0, self._number_of_clusters):
                point_belonging_count[i] = 0
                for j in range(0, dimensionality):
                    point_belonging_sum[i][j] = 0

            # for every point
            for i in range(0, len(self._rows_pca)):
                # we calculate the distance to every centroid and take the minimum
                for j in range(0, self._number_of_clusters):
                    current_distance = self.distance(self._rows_pca[i], self.__centroids[j])
                    if current_distance < distance_to_cluster[i]:
                        changed_value = True
                        self._predicted_values[i] = j
                        distance_to_cluster[i] = current_distance

                # once we have the minimum, we parse it
                closest_cluster = self._predicted_values[i]
                point_belonging_count[closest_cluster] += 1
                for k in range(0, dimensionality):
                    point_belonging_sum[closest_cluster][k] += self._rows_pca[i][k]

            # calculate the average position for each cluster
            for i in range(0, self._number_of_clusters):
                for j in range(0, dimensionality):
                    if point_belonging_count[i] != 0:
                        point_belonging_sum[i][j] /= point_belonging_count[i]
                    else:
                        point_belonging_sum[i][j] = 0

            # store the coordinates and begin anew
            for i in range(0, self._number_of_clusters):
                for j in range(0, dimensionality):
                    self.__centroids[i][j] = point_belonging_sum[i][j]

            # move on to the next iteration
            iteration_count += 1

    def display_clusters(self):
        plot.scatter(self._rows_pca[:, 0], self._rows_pca[:, 1],
                     c=self._predicted_values, s=50, cmap='viridis')
        plot.scatter([row[0] for row in self.__centroids], [row[1] for row in self.__centroids],
                     c='black', s=300, alpha=0.6)
        plot.show()
