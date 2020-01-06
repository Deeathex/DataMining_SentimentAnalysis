from statistical.naive_bayes.naivebayes import NaiveBayes

if __name__ == '__main__':
    nv = NaiveBayes()

    count = 0
    filename = "newDataset.csv"
    dataset = nv.load_data(filename)
    model = nv.create_model(dataset)
    # input should have all data with exception of label
    label = nv.predict(model, input)
