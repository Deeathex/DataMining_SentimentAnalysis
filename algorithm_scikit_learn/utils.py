import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split


def print_results(accuracy, precision, recall, f1, algorithm):
    print(algorithm)
    print("Accuracy:", (sum(accuracy) / len(accuracy)) * 100)
    print("Precision:", (sum(precision) / len(precision)) * 100)
    print("Recall:", (sum(recall) / len(recall)) * 100)
    print("F1 score:", (sum(f1) / len(f1)) * 100)


def compute_metrics(test_y, pred_y):
    accuracy = accuracy_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y)
    precision = precision_score(test_y, pred_y)
    recall = recall_score(test_y, pred_y)

    return accuracy, precision, recall, f1


def data_preparing():
    df = pd.read_pickle("../data_processing/text_processing/data/features_norm_minmax.pkl")
    df = df.drop(columns=["Raw text", "Text", "Id", "Number of negative hashtags", "Number of positive hashtags"])
    df = df.replace("pozitive", 1)
    df = df.replace("negative", 0)
    train_x, test_x, train_y, test_y = train_test_split(
        df.drop(columns=['Label']),
        df["Label"],
        train_size=0.8,
        test_size=0.2,
        stratify=df["Label"],
        random_state=1)

    # print("Test dataset: ", collections.Counter(test_y.values))
    # print("Training dataset: ", collections.Counter(train_y.values))
    return train_x, test_x, train_y, test_y
