import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from operator import itemgetter
pio.renderers.default = "browser"
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class TreeNode(object):
    def __init__(self, index=None, child=[], entropy=0, depth=0):
        self.index = index
        self.child = child
        self.entropy = entropy
        self.depth = depth

        self.splitAttribute = None
        self.order = None
        self.classification = None

    def setFeature(self, splitAttribute, order):
        self.splitAttribute = splitAttribute
        self.order = order

    def setClassification(self, classification):
        self.classification = classification


def entropy(frequency):
    # calculates entropy
    fraction = frequency / float(sum(frequency))
    return -np.sum(fraction * np.log(fraction))


class ID3(object):
    def __init__(self, maxDepth=10, minSplit=2, minGain=0.2):
        self.root = None
        self.maxDepth = maxDepth
        self.minSplit = minSplit
        self.trainSize = 0
        self.minGain = minGain

    def fit(self, data, target):
        self.trainSize = range(len(data))
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.classifications = target.unique()

        index = self.trainSize
        self.root = TreeNode(index=index, entropy=self.calcEntropy(index), depth=0)
        queue = [self.root]

        while queue:
            node = queue.pop()
            if node.depth < self.maxDepth or node.entropy < self.minGain:
                node.child = self.split(node)
                if not node.child:  # leaf node
                    self._setClassification(node)
                queue += node.child

            else:
                self._setClassification(node)


    def calcEntropy(self, index):
        # Calculates Entropy of node at index
        index = [i for i in index]
        frequency = np.array(self.target[index].value_counts())
        return entropy(frequency)

    def _setClassification(self, node):
        # find classification for a node if it is a leaf
        targetIndex = [i for i in node.index]
        node.setClassification(self.target[targetIndex].mode()[0])  # most frequent classification

    def split(self, node):
        index = node.index
        bestIG = 0
        bestSplit = []
        bestAttribute = None
        order = None
        subData = self.data.iloc[index, :]
        for i, attribute in enumerate(self.attributes):
            values = self.data.iloc[index, i].unique().tolist()
            if len(values) == 1: continue  # entropy = 0
            splits = []
            for val in values:
                sub_index = subData.index[subData[attribute] == val].tolist()

                splits.append([sub_id for sub_id in sub_index])

            # prevents splitting if less than minimum number of splits
            if min(map(len, splits)) < self.minSplit:
                continue

            # Information Gain
            entropySums = 0
            for split in splits:
                entropySums += len(split) * self.calcEntropy(split) / len(index)
            infoGain = node.entropy - entropySums
            print('This is the System Entropy')
            print(node.entropy)
            print('This is the Information Gain')
            print(infoGain)
            if infoGain < self.minGain:
                continue
            if infoGain > bestIG:
                bestIG = infoGain
                bestSplit = splits
                bestAttribute = attribute
                order = values
        node.setFeature(bestAttribute, order)
        childNodes = [TreeNode(index=split,
                                entropy=self.calcEntropy(split), depth=node.depth + 1) for split in bestSplit]
        return childNodes

    def predict(self, test):
        size = len(test)
        classifications = [None] * size # empty list size of instances in testing
        for instance in range(size):
            x = test.iloc[instance, :]  # one instance

            # start from root and traverse if not a leaf
            node = self.root
            while node.child:
                node = node.child[node.order.index(x[node.splitAttribute])]
            classifications[instance] = node.classification

        return classifications
    def evaluation(self, predictions, actual):
        length = len(predictions)
        correct = 0
        for index in range(len(predictions)):
            if predictions[index] == actual[index]:
                correct += 1
            else:
                continue
        accuracy = (correct/length) * 100
        return accuracy

def ID3_algo(trainingSet, testingSet, maxDepth, minSplit, minGain):
    xTrain = trainingSet.iloc [:, :-1]
    yTrain = trainingSet.iloc[:, -1]

    xTest = testingSet.iloc[:, :-1]
    yActual = testingSet.iloc[:, -1]

    tree = ID3(maxDepth,minSplit,minGain)
    tree.fit(xTrain,yTrain)
    predictions = tree.predict(xTest)
    print('These are the Predictions')
    print(predictions)
    print('\n These are the Actual')
    print(yActual)
    accuracy = tree.evaluation(predictions,yActual)
    # print('This is the accuracy')
    # print(accuracy, "\n")
    return accuracy


def CrossValidation (data, k , algo, maxDepth, minSplit, minGain):
    columnHeader = data.columns.to_list()
    data = data.to_numpy()
    np.random.shuffle(data)

    folds = np.array_split(data,k)

    accuracyList = []
    for i in range(k):
        testingSet = folds[i][:,:]
        newFolds = np.row_stack(np.delete(folds,i,0))
        trainingSet = newFolds[:,:]

        # print(f'Fold {i + 1}')
        # print(f'Testing set size : {testingSet.shape}')
        # print(f'Training set size : {trainingSet.shape}\n')
        trainingSet = pd.DataFrame(trainingSet, columns = columnHeader )
        testingSet = pd.DataFrame(testingSet, columns = columnHeader)
        accuracy = algo(trainingSet,testingSet,maxDepth,minSplit,minGain)
        accuracyList.append(accuracy)
    averageAccuracy = sum(accuracyList)/len(accuracyList)
    print(f'5 Fold Cross Validation Average Accuracy : {averageAccuracy}%\n')
    return accuracyList


# Car Classification
carheader = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acceptability']
carDF = pd.read_csv('data/car.data', names = carheader)
carDF_80 = carDF.sample(frac=0.8)
carTuningDF = carDF.drop(carDF_80.index)

carTuningDF = carTuningDF.reset_index(drop=True)
carDF_80 = carDF_80.reset_index(drop=True)

# Pruning Tuning Cross Validation
depthTuning = [x for x in range(2, 12, 2)]
splitTuning = [x for x in range(2, 12, 2)]
gainTuning = [x/10 for x in range(2, 12, 2)]
numFolds = [1,2,3,4,5]


##### Demonstration
## DATA
# – Demonstrate the calculation of information gain, gain ratio
# – Provide sample outputs from one test set on one fold for a classification tree.
videoTrain = carTuningDF.sample(frac=0.2)
videoTest = carTuningDF.drop(videoTrain.index)
videoTest = videoTest.reset_index(drop=True)
videoTrain = videoTrain.reset_index(drop=True)
videoAccuracy = ID3_algo(videoTrain,videoTest, 10, 3, 0.4)

# – Demonstrate an example traversing a classification tree and a class label being assigned at the leaf.
videoAccuracy = ID3_algo(videoTrain,videoTest.iloc[7:11], 10, 3, 0.4)