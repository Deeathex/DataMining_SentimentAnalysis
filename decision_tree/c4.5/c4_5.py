import numpy as np
import config as config
class C4_5:
    def __init__(self, training, testing, possible_outputs, attributes):
        self.__training = training
        self.__testing = testing
        self.__possible_outputs = possible_outputs
        self.__attributes = attributes

    def getTraining(self):
        return self.__training

    def howManyForEveryClass(self):
        how_many_for_every_class = {}
        how_many_training_examples = 0
        for train in self.__training:
            expected_output = train[1]
            if expected_output in how_many_for_every_class.keys():
                how_many_for_every_class[expected_output] += 1
            else:
                how_many_for_every_class[expected_output] = 1
            how_many_training_examples += 1
        return [how_many_for_every_class, how_many_training_examples]

    def entropy(self):
        how_many_for_every_class, how_many_training_examples = self.howManyForEveryClass()
        result = 0.0
        for key in how_many_for_every_class.keys():
            aux = how_many_for_every_class[key]/how_many_training_examples
            result += -aux*np.log2(aux)
        return result

    def preprocess(self, attribute_index):
        how_many_for_every_value = {}
        for train in self.__training:
            value = train[0][attribute_index]
            output = train[1]
            if value in how_many_for_every_value.keys():
                how_many_for_every_value[value][output] += 1
                how_many_for_every_value[value]["counter"] += 1
            else:
                how_many_for_every_value[value] = {}
                for possible_output in self.__possible_outputs:
                    if possible_output == output:
                        how_many_for_every_value[value][possible_output] = 1
                    else:
                        how_many_for_every_value[value][possible_output] = 0
                how_many_for_every_value[value]["counter"] = 1

        return how_many_for_every_value

    def preprocessForContinuousAttribute(self, threshold, attribute_index):
        how_many_for_every_value = {"let": {"counter": 0}, "gt": {"counter": 0}}
        for train in self.__training:
            value = train[0][attribute_index]
            output = train[1]
            if value <= threshold:
                if output in how_many_for_every_value["let"].keys():
                    how_many_for_every_value["let"][output] += 1
                else:
                    how_many_for_every_value["let"][output] = 1
                how_many_for_every_value["let"]["counter"] += 1
            else:
                if output in how_many_for_every_value["gt"].keys():
                    how_many_for_every_value["gt"][output] += 1
                else:
                    how_many_for_every_value["gt"][output] = 1
                how_many_for_every_value["gt"]["counter"] += 1
        return how_many_for_every_value

    def prs(self, attribute_index):
        how_many_for_every_value = self.preprocess(attribute_index)
        entropyValues = {}
        for key in how_many_for_every_value.keys():
            temp = how_many_for_every_value[key]
            result = 0.0
            for tempKey in temp.keys():
                if not (tempKey == "counter"):
                    aux = temp[tempKey] / temp["counter"]
                    if not(aux == 0):
                        result += -aux * np.log2(aux)
            entropyValues[key] = {"counter": temp["counter"], "value": result}

        return entropyValues

    def splitInfoForSpecificValue(self, attribute_index, value):
        how_many_for_every_value = self.preprocess(attribute_index)
        temp = how_many_for_every_value[value]
        result = 0.0
        for key in temp.keys():
            if not(key == "counter"):
                result += -temp[key]/len(self.__training)
        return result

    def splitInfoForSpecificAttribute(self, attribute_index):
        prsValues = self.prs(attribute_index)
        result = 0.0
        for key in prsValues:
            aux = prsValues[key]["counter"]/(len(self.__training))
            result += -aux * np.log2(aux)
        return result

    def gainForSpecificAttribute(self, attribute_index, totalNoTrainingExamples):
        result = self.entropy()
        prsValues = self.prs(attribute_index)
        for key in prsValues:
            result += -(prsValues[key]["counter"]/totalNoTrainingExamples) * prsValues[key]["value"]
        return result

    def gainRatioForSpecificAttribute(self, attribute_index):
        totalNoTrainingExamples = len(self.__training)
        gain = self.gainForSpecificAttribute(attribute_index, totalNoTrainingExamples)
        split = self.splitInfoForSpecificAttribute(attribute_index)
        return [gain, gain/split]

    def entropyForContinuousAttribute_lte_gt(self, value, training, attribute_index):
        lte_gt = {"let": {"noValue": 0}, "gt": {"noValue": 0}}
        for tr in training:
            current_value = tr[0][attribute_index]
            expected_output = tr[1]
            if current_value <= value:
                if expected_output in lte_gt["let"]:
                    lte_gt["let"][expected_output] += 1
                else:
                    lte_gt["let"][expected_output] = 1
                lte_gt["let"]["noValue"] += 1
            else:
                if expected_output in lte_gt["gt"]:
                    lte_gt["gt"][expected_output] += 1
                else:
                    lte_gt["gt"][expected_output] = 1
                lte_gt["gt"]["noValue"] += 1

        entropy_d = {"let": 0, "gt": 0, "no_let": 0, "no_gt": 0}
        let = lte_gt["let"]
        for key in let.keys():
            if not(key == "noValue"):
                entropy_d["let"] += -let[key]/let["noValue"]*np.log2(let[key]/let["noValue"])

        gt = lte_gt["gt"]
        for key in gt.keys():
            if not(key == "noValue"):
                entropy_d["gt"] += -gt[key]/gt["noValue"]*np.log2(gt[key]/gt["noValue"])
        entropy_d["no_let"] = let["noValue"]
        entropy_d["no_gt"] = gt["noValue"]
        return entropy_d

    def gainForContinuousAttribute(self, entropy_d, totalNoTrainingExamples):
        result = self.entropy() - entropy_d["let"]*(entropy_d["no_let"]/totalNoTrainingExamples) - entropy_d["gt"]*(entropy_d["no_gt"]/totalNoTrainingExamples)
        return result

    def splitInfoForContinuousAttribute(self, entropy_d, totalNoTrainingExamples):
        result = 0
        if not(entropy_d["no_let"] == 0):
            result += - (entropy_d["no_let"] / totalNoTrainingExamples) * np.log2(entropy_d["no_let"] / totalNoTrainingExamples)
        if not(entropy_d["no_gt"] == 0):
            result += - (entropy_d["no_gt"]/totalNoTrainingExamples) * np.log2(entropy_d["no_gt"]/totalNoTrainingExamples)
        return result

    def gainRatioForContinuosAttribute(self, value, attribute_index, training):
        entropy_d = self.entropyForContinuousAttribute_lte_gt(value, training, attribute_index)
        gain = self.gainForContinuousAttribute(entropy_d, len(training))
        split = self.splitInfoForContinuousAttribute(entropy_d, len(training))
        if split == 0.0:
            return [gain, 0.0]
        return [gain, gain/split]

    def findThresholdForContinuousAttribute(self, attribute_index):
        training = sorted(self.__training, key=
        lambda kv: kv[0][attribute_index])
        value = -1
        gainMax = -1
        gainRatio = -1
        for tr in training:
            val = tr[0][attribute_index]
            result = self.gainRatioForContinuosAttribute(val, attribute_index, training)
            gain = result[0]
            gainRatioAux = result[1]
            if value == -1:
                value = val
                gainMax = gain
                gainRatio = gainRatioAux
            elif gain > gainMax:
                gainMax = gain
                value = val
                gainRatio = gainRatioAux
        return [value, gainMax, gainRatio]

    def calculateGainAndGainRatio(self):
        gainAndGainRatios = []
        if self.__training == []:
            return
        exTr = self.__training[0][0]
        for i in range(0, len(exTr)):
            attribute = exTr[i]
            if config.ch[config.attr[i]] == 0:
                continue
            try:
                val = float(attribute)
                result = self.findThresholdForContinuousAttribute(i)
                result.append(self.__attributes[i])
                gainAndGainRatios.append(result)
            except ValueError:
                result = self.gainRatioForSpecificAttribute(i)
                result.append(self.__attributes[i])
                gainAndGainRatios.append(result)
        return gainAndGainRatios

    def myLambda(self, element):
        if len(element) == 3:
            return element[0]
        elif len(element) == 4:
            return element[1]

    def setTraining(self, training):
        self.__training = training

    def setAttributes(self, attributes):
        self.__attributes = attributes

    def getAtributes(self):
        return self.__attributes

    def findBest(self):
        result = self.calculateGainAndGainRatio()
        if result == None:
            return result
        gainAndGainRatios = sorted(result, key=
        lambda kv: self.myLambda(kv), reverse=True)
        if len(gainAndGainRatios) > 0:
            if len(gainAndGainRatios[0]) == 3:
                return gainAndGainRatios[0][2]
            else:
                return gainAndGainRatios[0][3]
        return None
