from algorithm_scikit_learn.utils import print_results, compute_metrics, data_preparing
from sklearn.tree import DecisionTreeClassifier


def run_decision_tree():
    train_x, test_x, train_y, test_y = data_preparing()
    model = DecisionTreeClassifier(random_state=0, criterion="entropy")
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    return compute_metrics(test_y, pred_y)


if __name__ == "__main__":
    accuracy = []
    precision = []
    f1 = []
    recall = []
    for i in range(0, 1000):
        a, p, r, f = run_decision_tree()
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    print_results(accuracy, precision, recall, f1, "Decision Tree")
