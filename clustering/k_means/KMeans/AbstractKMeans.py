from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot


class AbstractKMeans:
    def __init__(self, number_of_clusters, max_iterations=600):
        self._number_of_clusters = number_of_clusters
        self._max_iterations = max_iterations
        self._predicted_values = None
        self._rows_pca = None

    def display_elbow_method_graph(self, min_clusters=1, max_clusters=12):
        # calculate values for elbow method (find optimal number of clusters)
        number_of_clusters = range(min_clusters, max_clusters)
        elbow_kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_of_clusters]
        score = [elbow_kmeans[i].fit(self._rows_pca).score(self._rows_pca) for i in range(len(elbow_kmeans))]
        score = [i * -1 for i in score]

        # display the elbow method
        plot.plot(number_of_clusters, score)
        plot.xlabel('Number of Clusters')
        plot.ylabel('Score')
        plot.title('Elbow Method')
        plot.show()

    def scale_and_pca(self, processed_rows):
        # scale attributes
        scaler = StandardScaler()
        rows_scaled = scaler.fit_transform(processed_rows)

        # apply PCA (reduce multidimensional data to 2 dimensions)
        pca = IncrementalPCA(n_components=2, batch_size=200)
        self._rows_pca = pca.fit_transform(rows_scaled)

    def calculate_clusters(self):
        pass

    def display_clusters(self):
        pass

    def predict(self, predicted_data):
        pass

    def get_completeness_score(self, true_values):
        return completeness_score(true_values, self._predicted_values)

    def get_homogeneity_score(self, true_values):
        return homogeneity_score(true_values, self._predicted_values)

    def get_v_measure_score(self, true_values):
        return v_measure_score(true_values, self._predicted_values)

    def display_performance_metrics(self, true_values):
        print('Completeness score: ' + str(self.get_completeness_score(true_values)))
        print('Homogeneity score: ' + str(self.get_homogeneity_score(true_values)))
        print('V-measure score: ' + str(self.get_v_measure_score(true_values)))

    @staticmethod
    def get_score(true_values, predicted_values):
        correct_value_count = 0
        for i in range(0, len(true_values)):
            if true_values[i] == predicted_values[i]:
                correct_value_count += 1
        return correct_value_count / len(true_values)
