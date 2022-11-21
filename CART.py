# CART
# - only produces binary tree: non-leaf nodes always have two children

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from operator import itemgetter
pio.renderers.default = "browser"
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class CART:
    def __init__(self, tree='reg', criterion='mse', prune='depth', maxDepth=4, minCriterion=0.05):
        """
        :param tree: str - type of tree to be used 'reg' for regression or 'class' for regression
        :param criterion: str - 'mse' Mean Square Error, 'rsme' Root Mean Square Error
                                'entropy' for Entropy
        :param prune: str - type of pruning 'depth' and 'criterion'
        :param maxDepth: int - Max depth
        :param minCriterion: float - minimum criterion value
        """
        self.tree = tree
        self.criterion = criterion
        self.prune = prune
        self.maxDepth = maxDepth
        self.minCriterion = minCriterion

        self.attribute = None
        self.classification = None
        self.numSamples = None
        self.gain = None
        self.left = None
        self.right = None
        self.threshold = None
        self.root = None
        self.depth = 0

    def calcEvaluation(self, criterion, predictions, yActual):
        if criterion == 'mse':  # Mean Square Error
            return np.mean((predictions - yActual) ** 2.0)
        if criterion == 'rmse':  # Root MSe
            return (np.mean((predictions - yActual) ** 2.0)) ** 0.5

    def calcCriterion(self, criterion, target):

        if criterion == 'mse':  # Mean Square Error
            return np.mean((target - np.mean(target)) ** 2.0)

        elif criterion == 'rsme':  # root MSE
            return (np.mean((target - np.mean(target)) ** 2.0)) ** 0.5

        elif criterion == 'entropy':  # in the case of entropy
            entropy = 0.0
            for c in np.unique(target):
                # print ('this is the c',c)
                # print ('this is the target.shape' , target.shape[0])
                fraction = float(len(target[target == c])) / target.shape[0]
                if fraction > 0.0:
                    entropy -= fraction * np.log2(fraction)  # double check for cases that are not just binary outputs
            return entropy
        else:
            raise Exception('Criterion value is invalid')

    def buildTree(self, attributes, target, criterion):
        self.numSamples = attributes.shape[0]

        if len(np.unique(target)) == 1:
            self.classification = target[0]
            return

        bestGain = 0.0
        bestFeature = None
        bestThreshold = None

        if criterion in {'mse', 'rsme'}:
            self.classification = np.mean(target)

        elif criterion == 'entropy':
            self.classification = max([(c, len(target[target == c])) for c in np.unique(target)], key=lambda x: x[1])[0]

        else:
            raise Exception('Criterion value is invalid')

        criterionValue = self.calcCriterion(criterion, target)

        for column in range(attributes.shape[1]):
            attributeLevel = np.unique(attributes[:, column])
            # print ('This is the attribute level', attributeLevel)
            thresholds = (attributeLevel[:-1] + attributeLevel[1:]) / 2.0
            # print('this is the threshold')
            # print(thresholds)

            for threshold in thresholds:
                targetLeft = target[attributes[:, column] <= threshold]
                criterionLeft = self.calcCriterion(criterion, targetLeft)
                numsLeft = float(targetLeft.shape[0]) / self.numSamples

                targetRight = target[attributes[:, column] > threshold]
                criterionRight = self.calcCriterion(criterion, targetRight)
                numsRight = float(targetRight.shape[0]) / self.numSamples

                currentGain = criterionValue - (numsLeft * criterionLeft + numsRight * criterionRight)
                if currentGain > bestGain:
                    bestGain = currentGain
                    bestFeature = column
                    bestThreshold = threshold

        self.gain = bestGain
        self.attribute = bestFeature
        self.threshold = bestThreshold
        self.split(attributes, target, criterion)

    def split(self, attributes, target, criterion):
        if self.threshold is None:
            return
        # Left Split
        attributesLeft = attributes[attributes[:, self.attribute] <= self.threshold]
        targetLeft = target[attributes[:, self.attribute] <= self.threshold]
        self.left = CART()
        self.left.depth = self.depth + 1
        self.left.buildTree(attributesLeft, targetLeft, criterion)

        # Right Split
        attributesRight = attributes[attributes[:, self.attribute] > self.threshold]
        targetRight = target[attributes[:, self.attribute] > self.threshold]
        self.right = CART()
        self.right.depth = self.depth + 1
        self.right.buildTree(attributesRight, targetRight, criterion)

    def pruningTree(self, method, maxDepth, minCriterion, numSamples):
        if self.attribute is None:
            return

        self.left.pruningTree(method, maxDepth, minCriterion, numSamples)
        self.right.pruningTree(method, maxDepth, minCriterion, numSamples)

        pruning = False

        # pruning based on depth
        if method == 'depth' and self.depth >= maxDepth:
            pruning = True
            print('We Are Going to Pruning')

        # pruning based on criterion value
        elif method == 'criterion' and self.left.attribute is None and self.right.attribute is None:
            if (self.gain * float(self.numSamples) / numSamples) < minCriterion:
                pruning = True
                print('We Are Going to Stop Early')
            elif minCriterion == 0:  # case where the criterion is 0 then it won't stop pruning
                pruning = False
                print('We are Not going to Stop Early')

        else:
            print('We are going to stop pruning')

        if pruning is True:
            self.left = None
            self.right = None
            self.attribute = None

    def predict(self, instance):
        if self.attribute != None:
            if instance[self.attribute] <= self.threshold:
                return self.left.predict(instance)
            else:
                return self.right.predict(instance)
        else:
            return self.classification

    def showTree(self, depth, cond):
        base = '    ' * depth + cond
        if self.attribute != None:
            print(base + 'if X[' + str(self.attribute) + '] <= ' + str(self.threshold))
            self.left.showTree(depth + 1, 'then ')
            self.right.showTree(depth + 1, 'else ')
        else:
            print(base + '{value: ' + str(self.classification) + ', samples: ' + str(self.numSamples) + '}')

    def fit(self, attributes, target):
        self.root = CART()
        if (self.tree == 'reg'):
            self.root.buildTree(attributes, target, self.criterion)
        elif (self.tree == 'class'):
            self.root.buildTree(attributes, target, self.criterion)
        else:
            raise Exception('Tree type is invalid')
        self.root.pruningTree(self.prune, self.maxDepth, self.minCriterion, self.root.numSamples)

    def prediction(self, attributes):
        return np.array([self.root.predict(f) for f in attributes], dtype='object')

    def printTree(self):
        self.root.showTree(0, ' ')


def CART_algo(trainingSet, testingSet, tree, criterion, prune, maxDepth, minCriterion):
    xTrain = trainingSet[:, :-1]
    yTrain = trainingSet[:, -1]

    xTest = testingSet[:, :-1]
    yActual = testingSet[:, -1]
    # print('This is xTrain')
    # print(xTrain)
    # print(xTrain.shape)
    # print('This is yActual')
    # print(yActual)
    # print(yActual.shape)
    tree = CART(tree, criterion, prune, maxDepth, minCriterion)
    print('Regression Tree:\n')
    tree.fit(xTrain, yTrain)
    tree.printTree()
    pred = tree.prediction(xTest)
    # print('this is the predictions')
    # print(pred.shape)
    evaluation = tree.calcEvaluation(criterion, pred, yActual)
    print('\nThis is the evaluation for', criterion + ':')
    print(evaluation, "\n")
    return pred, yActual, evaluation


def CrossValidation(data, k, algo, tree, criterion, prune, maxDepth, minCriterion):
    np.random.shuffle(data)
    folds = np.array_split(data, k)
    evaluationList = []
    predictions = []
    actuals = []
    for i in range(k):
        testingSet = folds[i][:, :]
        newFolds = np.row_stack(np.delete(folds, i, 0))
        trainingSet = newFolds[:, :]

        print(f'Fold {i + 1}')
        # print(f'Testing set size : {testingSet.shape}')
        # print(f'Training set size : {trainingSet.shape}\n')
        pred, yActual, evaluation = algo(trainingSet, testingSet, tree, criterion, prune, maxDepth, minCriterion)
        evaluationList.append(evaluation)
        predictions.append(pred)
        actuals.append(yActual)
    print('This is the Average Evaluation for the Cross Validation\n')
    print(sum(evaluationList)/len(evaluationList))
    return evaluation, evaluationList, pred, yActual


###################################
# # Forest Fires
forestDF = pd.read_csv('data/forestfires.data', header=0)
forestDF = forestDF[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH',
                'wind', 'rain', 'area']]

forestDF['month'] = pd.to_datetime(forestDF['month'], format='%b').dt.month
weekday_map = {'mon':1.0,'tue':2.0,'wed':3.0,'thu':4.0,'fri':5.0,'sat':6.0,'sun':7.0}
forestDF = forestDF.replace({'day':weekday_map})
forestDF = forestDF.astype(float)
print('Forest Fire Regression Problem \n')
# print(forestDF.head())

forestDF_80 = forestDF.sample(frac=0.8)
forestTuningDF = forestDF.drop(forestDF_80.index)
# print('This is the size of the original:', len(forestDF))
# print('This is the size of the 80%:', len(forestDF_80))
# print('This is the size of the 20%:', len(forestTuningDF), '\n')

forestTuningNP = forestTuningDF.to_numpy()
forestDF_80 = forestDF_80.to_numpy()

#Testing Pruned and NonPruned Trees
# numFolds = [1,2,3,4,5]
# print('Pruned Tree Results')
# ForestTestingPrunedResults, ForestTestingPrunedList = CrossValidation(forestDF_80, 5, CART_algo,tree='reg', criterion='mse', prune='criterion', maxDepth=90, minCriterion=2.5)
# print('Unpruned Tree Results')
# ForestTestingNPResults, ForestTestingNPList = CrossValidation(forestDF_80, 5, CART_algo,tree='reg', criterion='mse', prune='criterion', maxDepth=0, minCriterion=2.5)
#
# ForestDict = {'Folds': numFolds, 'Pruned':ForestTestingPrunedList,'Non-Pruned':ForestTestingNPList}
#
# finalForestResults = pd.DataFrame(ForestDict)
#
# ForestFinalFig = px.scatter(finalForestResults, x='Folds', y=['Pruned', 'Non-Pruned'] , title="K-Fold vs. MSE", width=800, height=600)
# ForestFinalFig.update_traces(marker_size = 15)
# ForestFinalFig.update_yaxes(title_text = 'MSE')
# ForestFinalFig.show()

###### Video Demonstration
# DATA FOR VIDEO
forestVideoTrain = forestTuningDF.sample(frac=0.3)
forestVideoTest = forestTuningDF.drop(forestVideoTrain.index)
forestVideoTrain = forestVideoTrain.to_numpy()
forestVideoTest = forestVideoTest.to_numpy()



# – Show a sample regression tree without early stopping and with early stopping.

# print('Without Stopping Early')
# CART_algo(forestVideoTrain,forestVideoTest,tree='reg', criterion='mse', prune='criterion',
#                                                          maxDepth=0, minCriterion=0)
#
# print('Stopping Early')
# CART_algo(forestVideoTrain,forestVideoTest,tree='reg', criterion='mse', prune='criterion',
#                                                          maxDepth=4, minCriterion=2.5)


# – Provide sample outputs from one test set on one fold for a regression tree.

videoPredictions, videoActuals, videoEvaluation = CART_algo(forestVideoTrain,forestVideoTest,tree='reg', criterion='mse', prune='criterion',
                                                         maxDepth=20, minCriterion=2.5)
print('This is the Actual Values for Forest Fires')
print(videoActuals)

print('\n This is the Prediction Values for Forest Fires')
print(videoPredictions)

print('\n This is the Evaluation Values for Forest Fires')
print(videoEvaluation)


# – Demonstrate the calculation mean squared error.
test = CART()
mse = test.calcEvaluation(criterion='mse', predictions=videoPredictions, yActual=videoActuals)
print('\n This is the MSE')
print(mse)

# – Demonstrate a decision being made to prune a subtree (pruning) and a decision being made to
# stop growing a subtree (early stopping).

CART_algo(forestVideoTrain,forestVideoTest,tree='reg', criterion='mse', prune='criterion',
                                                         maxDepth=20, minCriterion=0)

# – Demonstrate an example traversing a regression tree and a prediction being made at the leaf.

trainX =forestVideoTrain[:,:-1]
trainY = forestVideoTrain[:,-1]
testX = forestVideoTest[4:10,:-1]
testY = forestVideoTest[4:10,-1]

test.fit(trainX,trainY)

prediction = test.prediction(testX)
print('This is the Prediction')
print(prediction)
print('\n This is the Actuals')
print(testY)


