from csv import reader
from math import exp
from math import pi
from math import sqrt

class NaiveBayes:

    def load_data(self, filename):
        dataset = []
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)

        for i in range(len(dataset[0]) - 1):
            self.__str_column_to_float(dataset, i)
        # convert classes to integers
        self.__str_column_to_int(dataset, len(dataset[0]) - 1)
        return dataset


    def __str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())


    def __str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = {}
        for i, value in enumerate(unique):
            lookup[value] = i
            print('[%s] => %d' % (value, i))
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup


    # Split the dataset by class values, returns a dictionary
    def __separate_by_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if class_value not in separated:
                separated[class_value] = []
            separated[class_value].append(vector)
        return separated


    def __mean(self, numbers):
        return sum(numbers) / float(len(numbers))


    def __stdev(self, numbers):
        avg = self.__mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance)


    # Calculate the mean, stdev and count for each column in a dataset
    def __summarize_dataset(self, dataset):
        summaries = [(self.__mean(column), self.__stdev(column), len(column)) for column in zip(*dataset)]
        summaries.pop(-1)
        return summaries


    # Split dataset by class then calculate statistics for each row
    def create_model(self, dataset):
        separated = self.__separate_by_class(dataset)
        model = {}
        for class_value, rows in separated.items():
            model[class_value] = self.__summarize_dataset(rows)
        return model


    # Calculate the Gaussian probability distribution function for x
    def __calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent


    # Calculate the probabilities of predicting each class for a given row
    def __calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.__calculate_probability(row[i], mean, stdev)
        return probabilities


    # Predict the class for a given row
    def predict(self, model, row):
        probabilities = self.__calculate_class_probabilities(model, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

