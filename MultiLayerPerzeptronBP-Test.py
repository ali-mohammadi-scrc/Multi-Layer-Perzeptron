from MultiLayerPerzeptronBP import MultiLayerPerzeptronBP
from MultiLayerPerzeptronBP import gnuplotOut
import random as rand

#### Example 1
print('Training model for XOR')
E, XOR = MultiLayerPerzeptronBP ([2, 2, 1], [1, 2], [], [([0, 0], [0]), ([1, 0], [1]), ([0, 1], [1]), ([1, 1], [0])], [0.1, 0.01], 17, 5000, 0)
print('XOR([0, 0]): ' + str(XOR([0, 0])))
print('XOR([0, 1]): ' + str(XOR([0, 1])))
print('XOR([1, 0]): ' + str(XOR([1, 0])))
print('XOR([1, 1]): ' + str(XOR([1, 1])))
gnuplotOut('XOR_learning.curve', list(range(len(E))), E, 'No. Patterns', 'Error', 'XOR Learning Curve')

#### Example 2
print('Training model for y_1 = 0.35 * x_1 - 0.83 * x_2, y_2 = x_1 ^ 2 for X_1, X_2 in [-3, 3]')
f = lambda x: [0.35  * x[0] - 0.83 * x[1], x[1] ** 2]
NTrainingPatterns = 200;
NTestPatterns = 10;
X = [[rand.random() * 6 - 3 for j in range(2)] for i in range(NTrainingPatterns)]
TrainingPatterns = [(x, f(x)) for x in X]
X = [[rand.random() * 6 - 3 for j in range(2)] for i in range(NTestPatterns)]
TestPatterns = [(x, f(x)) for x in X]
Layers = [2, 3, 2, 2]
ActivationFunctions = [0, 1, 2]
Weights = []
LearningRates = [0.1, 0.1, 0.01]
RandomSeed = 17
MaxSteps = 250
Batch = 0
E, F = MultiLayerPerzeptronBP (Layers, ActivationFunctions, Weights, TrainingPatterns, LearningRates, RandomSeed, MaxSteps, Batch)

TestErr = [sum([(a - b) ** 2 for a, b in zip(p[1], F(p[0]))]) for p in TestPatterns]
print('Total error for test patterns: ' + str(sum(TestErr)))
for i in range(len(TestErr)):
    x = TestPatterns[i][0]
    y = TestPatterns[i][1]
    print('Pattern#' + str(i) + ' X = ' + str(x) + ' - Y = ' + str(y) + ' -  Y_model = ' + str(F(x)) + ' - Error: ' + str(TestErr[i]))
gnuplotOut('learning.curve', list(range(len(E))), E, 'No. Patterns', 'Error', 'Learning Curve')
