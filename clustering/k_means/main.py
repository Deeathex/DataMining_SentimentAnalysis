from clustering.k_means.KMeans.SklearnKMeans import SklearnKMeans
from clustering.k_means.Dataset.ProjectDataset import ProjectDataset
from clustering.k_means.KMeans.MyKMeans import MyKMeans

import pandas as pd
import numpy as np

path_to_csv = "C:/datamining/newDataset.csv"


def add_metrics(average_list, max_list, k_means, true_values):
    # calculate metrics
    completeness_score = k_means.get_completeness_score(true_values)
    homogeneity_score = k_means.get_homogeneity_score(true_values)
    v_measure_score = k_means.get_v_measure_score(true_values)

    # average
    average_list[0] += completeness_score
    average_list[1] += homogeneity_score
    average_list[2] += v_measure_score

    # max
    if max_list[2] < v_measure_score:
        max_list[0] = completeness_score
        max_list[1] = homogeneity_score
        max_list[2] = v_measure_score


def execute_k_means(k_means, processed_rows):
    k_means.scale_and_pca(processed_rows)
    k_means.calculate_clusters()


def main():
    # read our dataset and get our required values (values & class)
    dataset = ProjectDataset(path_to_csv)
    processed_rows = dataset.get_preprocessed_rows_values()
    processed_class = dataset.get_preprocessed_rows_class()
    dataset.display_data_metrics()

    # initialize the average values and the iteration count
    iteration_count = 100
    split_data_train_test = True
    average_sklearn = [0, 0, 0]
    average_my_euclidean = [0, 0, 0]
    average_my_manhattan = [0, 0, 0]
    average_my_minkowski = [0, 0, 0]
    max_sklearn = [0, 0, 0]
    max_my_euclidean = [0, 0, 0]
    max_my_manhattan = [0, 0, 0]
    max_my_minkowski = [0, 0, 0]
    distances = ["euclidean", "manhattan", "minkowski"]
    averages = [average_my_euclidean, average_my_manhattan, average_my_minkowski]
    max_values = [max_my_euclidean, max_my_manhattan, max_my_minkowski]

    # separate into training and testing sets
    train_processed_rows = processed_rows[:1800]
    train_processed_class = processed_class[:1800]

    test_processed_rows = processed_rows[1800:]
    test_processed_class = processed_class[1800:]

    best_value_sklearn = 0
    best_value_my_distance = [0, 0, 0]
    average_accuracy_sklearn = 0
    average_accuracy_my_distance = [0, 0, 0]

    # start the computation
    for i in range(0, iteration_count):
        print('Current iteration: ' + str(i + 1) + "/" + str(iteration_count))
        # sklearn
        k_means = SklearnKMeans(number_of_clusters=3, max_iterations=600)
        execute_k_means(k_means, train_processed_rows if split_data_train_test else processed_rows)
        add_metrics(average_sklearn, max_sklearn, k_means, train_processed_class if split_data_train_test else processed_class)
        if split_data_train_test:
            test_predicted_values = k_means.predict(test_processed_rows)
            current_score = k_means.get_score(test_processed_class, test_predicted_values)
            average_accuracy_sklearn += current_score
            if current_score > best_value_sklearn:
                best_value_sklearn = current_score

        # myKMeans
        for j in range(0, len(distances)):
            k_means = MyKMeans(number_of_clusters=3, max_iterations=600, distance_type=distances[j])
            execute_k_means(k_means, train_processed_rows if split_data_train_test else processed_rows)
            add_metrics(averages[j], max_values[j], k_means, train_processed_class if split_data_train_test else processed_class)
            if split_data_train_test:
                test_predicted_values = k_means.predict(test_processed_rows)
                current_score = k_means.get_score(test_processed_class, test_predicted_values)
                average_accuracy_my_distance[j] += current_score
                if current_score > best_value_my_distance[j]:
                    best_value_my_distance[j] = current_score

    # calculate the actual averages
    average_sklearn = [x / iteration_count for x in average_sklearn]
    for i in range(0, len(averages)):
        averages[i] = [x / iteration_count for x in averages[i]]
    if split_data_train_test:
        average_accuracy_sklearn = average_accuracy_sklearn / iteration_count
        for j in range(0, len(distances)):
            average_accuracy_my_distance[j] = average_accuracy_my_distance[j] / iteration_count

    # do some preprocessing on the names of the metrics
    distances += ["sklearn"]
    measures = ["completeness", "homogeneity", "v-measure"]

    # print the averages
    averages += [average_sklearn]
    averages = np.array(averages).transpose()
    data_frame = pd.DataFrame(averages, columns=distances, index=measures)
    print("Average values, " + str(iteration_count) + " iterations:")
    print(data_frame)

    # print the max values
    max_values += [max_sklearn]
    max_values = np.array(max_values).transpose()
    data_frame = pd.DataFrame(max_values, columns=distances, index=measures)
    print()
    print("Best values, " + str(iteration_count) + " iterations:")
    print(data_frame)

    # print the predicted values if we have them
    if split_data_train_test:
        average_values = np.array(average_accuracy_my_distance + [average_accuracy_sklearn]).transpose()
        data_frame = pd.DataFrame(average_values, index=distances, columns=["accuracy"])
        print()
        print("Average values, " + str(iteration_count) + " iterations:")
        print(data_frame)

        best_values = np.array(best_value_my_distance + [best_value_sklearn]).transpose()
        data_frame = pd.DataFrame(best_values, index=distances, columns=["accuracy"])
        print()
        print("Best values, " + str(iteration_count) + " iterations:")
        print(data_frame)


if __name__ == '__main__':
    # run the k-means
    main()
