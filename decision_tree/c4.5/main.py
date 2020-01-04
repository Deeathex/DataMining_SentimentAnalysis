from c4_5 import *
from tree import *
import config as config
import csv

c = None
# specific_values_for_attributes = {"outlook": ["sunny", "overcast", "rain"], "wind": ["weak", "strong"],
# 									  "temperature": c.findThresholdForContinuousAttribute(1),
# 									  "humidity": c.findThresholdForContinuousAttribute(2)}
specific_values_for_attributes = {}


queue = []

def read(splitIndex):
    with open('newDataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        config.training = []
        for row in csv_reader:
            if line_count == 0:
                index = 0
                config.attributes = row[:len(row) - 1]
                config.ch = {}
                config.attr = {}
                config.continuous = {}
                config.attributes_index = {}
                for el in config.attributes:
                    config.ch[el] = 1
                    config.attributes_index[el] = index
                    config.attr[index] = el
                    config.continuous[el] = 1
                    index += 1
                line_count = 1
            else:
                entity = []
                line = row[:len(row) - 1]
                label = row[len(row) - 1]
                for el in line:
                    try:
                        el = float(el)
                        entity.append(el)
                    except:
                        entity.append(0.0)
                if line_count <= splitIndex:
                    config.training.append([entity, label])
                else:
                    config.testing.append([entity, label])
                line_count += 1

def trainingSetForEveryValueForSpecificAttribute(values, index, fromL):
    result = []
    for value in values:
        trainingsWithSpecificValue = [[t[0], t[1]] for t in fromL if t[0][index] == value]
        result.append(trainingsWithSpecificValue)
    return result

def testing(root):
    noTests = 0
    noCorrect = 0
    for test in config.testing:
        v = verifyTest(root, test[0])
        if v == test[1]:
            noCorrect += 1
        noTests += 1

    print (noCorrect/noTests * 100)

def main_c4_52():
    best = c.findBest()
    config.ch[best] = 0
    first = Node(None, None, best)
    c4_52(first, best)
    testing(first)


def trainingSetForEveryValueForSpecificAttributeContinuous(value, index, fromL):
    result = []
    if isinstance(value, list):
        trainingslet = [[t[0], t[1]] for t in fromL if t[0][index] <= value[0]]
        trainingsgt = [[t[0], t[1]] for t in fromL if t[0][index] > value[0]]
    else:
        trainingslet = [[t[0], t[1]] for t in fromL if t[0][index] <= value]
        trainingsgt = [[t[0], t[1]] for t in fromL if t[0][index] > value]
    result.append(trainingslet)
    result.append(trainingsgt)
    return result

def reinitialize():
    index_attribute = 0
    c.setTraining(config.training)
    for attribute in config.attributes:
        specific_values_for_attributes[attribute] = c.findThresholdForContinuousAttribute(index_attribute)
        index_attribute += 1

def c4_52(current, best):
    next_level = [best]
    queue = [best]
    ok = 0
    temp = c.getTraining()
    while 1:
        if ok == 0:
            queue.pop(0)
            ok = 1
        reinitialize()
        specific_values = specific_values_for_attributes[best]
        if isinstance(specific_values, list):
            i = 0
            if config.continuous[best] == 1:
                k = 0
                result = trainingSetForEveryValueForSpecificAttributeContinuous(specific_values, config.attributes_index[best], temp)
                for sett in result:
                    if not (sett == []):
                        c.setTraining(sett)
                        r = c.howManyForEveryClass()
                        if len(r[0].keys()) == 1:
                            keys = list(r[0].keys())
                            current.addChild(Node(keys[0], specific_values[k] + i, None))
                        else:
                            current.addChild(Node(None, specific_values[k] + i, None, sett))
                        i += 0.00001
            for node in current.getChildren():
                if node.getOutput() == None:
                    try:
                        temp = node.getList()
                        val = float(node.getEdge())
                        aux = trainingSetForEveryValueForSpecificAttributeContinuous(node.getEdge(), config.attributes_index[best], temp)
                        if aux[0] == []:
                            temp = aux[1]
                        else:
                            temp = aux[0]
                        if not (temp == []):
                            c.setTraining(temp)
                            best1 = c.findBest()
                            if best1 == None:
                                node.setOutput(config.labels[0])
                                continue
                            newNode = Node(None, node.getEdge(), best1)
                            node.addChild(newNode)
                            config.ch[best1] = 0
                            next_level.append(best1)
                            queue.append([newNode, temp[:]])
                    except ValueError:
                        aux = trainingSetForEveryValueForSpecificAttribute([node.getEdge()], config.attributes_index[best], temp)
                        temp = aux[0]
                        if not (temp == []):
                            c.setTraining(temp)
                            best1 = c.findBest()
                            if best1 == None:
                                node.setOutput(config.labels[0])
                                continue
                            newNode = Node(None, node.getEdge(), best1)
                            node.addChild(newNode)
                            config.ch[best1] = 0
                            next_level.append(best1)
                            queue.append([newNode, aux[0][:]])
        try:
            next_level.pop(0)
            best = next_level[0]
            print(best)
            if best == None:
                break
            current, config.training = queue[0]
            queue.pop(0)
        except IndexError:
            break

def verifyTest(node, test):
    if node.getOutput():
        return node.getOutput()
    if node.getName():
        index = config.attributes_index[node.getName()]
    else:
        if node.getEdge() in config.attributes_index.keys():
            index = config.attributes_index[node.getEdge()]
        else:
            return verifyTest(node.getChildren()[0], test)
    try:
        value = float(test[index])
        if value <= node.getChildren()[0].getEdge():
            return verifyTest(node.getChildren()[0], test)
        else:
            if(len(node.getChildren())>1):
                return verifyTest(node.getChildren()[1], test)
            else:
                return verifyTest(node.getChildren()[0], test)
    except ValueError:
        value = test[index]
        for n in node.getChildren():
            if n.getEdge() == value:
                return verifyTest(n, test)

if __name__ == "__main__":
    noTrainingEntities = int(input("Number of training entites:"))
    read(noTrainingEntities)
    c = C4_5(config.training, [], config.labels, config.attributes)
    main_c4_52()