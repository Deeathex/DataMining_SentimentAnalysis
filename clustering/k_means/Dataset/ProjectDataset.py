import pandas as pd


class ProjectDataset:
    def __init__(self, path_to_csv):
        self.__path_to_csv = path_to_csv
        self._raw_data = None
        self.__read_csv()

    def __read_csv(self):
        # display all columns from the dataset if ever printed
        pd.options.display.width = 0

        # read the data from the pkl file
        self._raw_data = pd.read_csv(self.__path_to_csv)

    def get_preprocessed_rows_values(self):
        rows_values = self._raw_data.iloc[:, :-1].values
        return rows_values

    def get_preprocessed_rows_class(self):
        rows_class = self._raw_data.iloc[:, -1].values

        # we do some preprocessing on the values (string to numerical values)
        for i in range(0, len(rows_class)):
            if rows_class[i] == "negative":
                rows_class[i] = 0
            if rows_class[i] == "pozitive":
                rows_class[i] = 1

        return rows_class

    def display_data_metrics(self):
        # print the first 5 lines
        print(self.get_preprocessed_rows_values()[:5, :])

        # print the array shape: (nr. of rows, nr of columns)
        print(self._raw_data.shape)
