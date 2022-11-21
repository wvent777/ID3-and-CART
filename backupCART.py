# CART
# - only produces binary tree: non-leaf nodes always have two children

import pandas as pd
import numpy as np

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

        # pruning based on criterion value
        elif method == 'criterion' and self.left.attribute is None and self.right.attribute is None:
            if (self.gain * float(self.numSamples) / numSamples) < minCriterion:
                pruning = True

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



def CART_algo(trainingSet, testingSet, tree, criterion, prune, maxDepth):
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
    tree = CART(tree, criterion, prune, maxDepth)
    print('Regression Tree of max depth',maxDepth,'\n')
    tree.fit(xTrain, yTrain)
    tree.printTree()
    pred = tree.prediction(xTest)
    # print('this is the predictions')
    # print(pred.shape)
    evaluation = tree.calcEvaluation(criterion, pred, yActual)
    print('\nThis is the evaluation for', criterion+':')
    print(evaluation,"\n")
    return pred, yActual

def CrossValidation(data, k, algo, tree, criterion, prune, maxDepth):
    folds = np.array_split(data, k)
    np.random.shuffle(data)
    folds = np.array_split(data, k)

    for i in range(k):
        testingSet = folds[i][:, :]
        newFolds = np.row_stack(np.delete(folds, i, 0))
        trainingSet = newFolds[:, :]

        # some print functions
        print(f'Fold {i + 1}')
        print(f'Testing set size : {testingSet.shape}')
        print(f'Training set size : {trainingSet.shape}\n')

        pred, yActual = algo(trainingSet, testingSet, tree, criterion, prune, maxDepth)


####################################
# Regression Problems Abalone , computer hardware , forest fires
# Abalone Regression Problem
abaloneheader = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight'
    , 'Shell Weight', 'Rings']

abaloneDF = pd.read_csv('data/abalone.data', names=abaloneheader)
print(abaloneDF.head())
abaloneDF = abaloneDF[['Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']]
abaloneDF_80 = abaloneDF.sample(frac=0.8)
abaloneTuningDF = abaloneDF.drop(abaloneDF_80.index)

print('This is the size of the original', len(abaloneDF))
print('This is the size of the 80%', len(abaloneDF_80))
print('This is the size of the 20%', len(abaloneTuningDF))

abaloneTuningNP = abaloneTuningDF.to_numpy()
print('This is the numpy version')
print(abaloneTuningNP.shape)
print(abaloneTuningNP.ndim)
print(type(abaloneTuningNP))




# CrossValidation(abaloneTuningNP, 3, CART_algo, tree='reg', criterion='mse', prune='depth', maxDepth=3)

carheader = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','acceptability']

carDF = pd.read_csv('data/car.data', names=carheader, index_col=False)
carDF = carDF[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','acceptability']]

print(carDF.head())
carDF_80 = carDF.sample(frac=0.8)
carTuningDF = carDF.drop(carDF_80.index)

print('This is the size of the original:', len(carDF))
print('This is the size of the 80%:', len(carDF_80))
print('This is the size of the 20%:', len(carTuningDF), '\n')

carTuningNP = carTuningDF.to_numpy()
carDF_80 = carDF_80.to_numpy()

# Cross Validation for the tuning on mse criterion
CrossValidation(carTuningNP, 2, CART_algo, tree='class', criterion='entropy', prune='criterion', maxDepth=3, minCriterion=0.2)