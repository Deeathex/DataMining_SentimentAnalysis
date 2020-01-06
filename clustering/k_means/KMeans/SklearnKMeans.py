from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from clustering.k_means.KMeans.AbstractKMeans import AbstractKMeans

from sklearn.cluster import KMeans
import matplotlib.pyplot as plot


class SklearnKMeans(AbstractKMeans):
    def __init__(self, number_of_clusters, max_iterations=600):
        super().__init__(number_of_clusters, max_iterations)
        self.__k_means = None

    def calculate_clusters(self):
        # run the KMeans algorithm
        self.__k_means = KMeans(n_clusters=self._number_of_clusters, max_iter=self._max_iterations, algorithm='auto')
        self._predicted_values = self.__k_means.fit_predict(self._rows_pca)
        return self._predicted_values

    def predict(self, predicted_data):
        # scale attributes
        scaler = StandardScaler()
        rows_scaled = scaler.fit_transform(predicted_data)

        # apply PCA (reduce multidimensional data to 2 dimensions)
        pca = IncrementalPCA(n_components=2, batch_size=200)
        rows_scaled = pca.fit_transform(rows_scaled)

        return self.__k_means.predict(rows_scaled)

    def display_clusters(self):
        plot.scatter(self._rows_pca[:, 0], self._rows_pca[:, 1], c=self._predicted_values, s=50, cmap='viridis')
        cluster_centers = self.__k_means.cluster_centers_
        plot.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=300, alpha=0.6)
        plot.show()
