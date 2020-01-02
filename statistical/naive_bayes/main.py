from statistical.naive_bayes.naivebayes import  NaiveBayes

if __name__ == '__main__':
    nv = NaiveBayes()

    count = 0
    filename = "filename.csv"
    dataset = nv.load_data(filename)
    model = nv.create_model(dataset)
    #input should have all data with expcetion of label
    label = nv.predict(model, input)
    
