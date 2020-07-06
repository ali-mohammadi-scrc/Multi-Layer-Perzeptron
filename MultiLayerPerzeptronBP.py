import random as rand
import math

def readNums(File): # to read numbers only from a file for weights and patterns initialization
    f = open(File, 'r')
    data = []
    for l in f:
        l = l[0:l.find('#')] # anything followed by a '#' will be ignored
        data = data + [float(x) for x in l.split()] # two numbers must be seprated with a whitespace
    f.close()
    return data

def readPatterns(File, N, M): # to read training and test patterns from a file using readNums function and form them in a desirable way for further process 
    data = readNums(File)
    P = int(len(data)/(N+M))
    return [([data[(p * (M + N)) + n] for n in range(N)], [data[(p * (M + N)) + N + m] for m in range(M)]) for p in range(P)]
    # patterns array, contains P tuples consist of 2 arrays, the first array is input and the secod one is desired output
    # eg. [([1, 1], [1])] is a pattern array with one pattern, in which [1, 1] is input and [1] is desired output
    
def MPNeuron (X, W, F): # A simple Neuron which computes net = weighted suum of inputs and pass it to an activation function
    return [F(sum([x * w for x, w in zip(X, ws)])) for ws in W]

def MultiLayerPerzeptronBP (Layers, ActivationFunctions, Weights, Patterns, LearningRates, RandomSeed, MaxSteps, Batch): # Please read the "Read Me" file 
    rand.seed(RandomSeed)
    ### Part 1 ###
    Tanh = lambda x: (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    dTanh = lambda y: 1 - y ** 2
    
    Logistic = lambda x: 1/(1 + math.exp(-x))
    dLogistic = lambda y: y * (1 - y)
    
    Identity = lambda x: x
    dIdentity = lambda y: 1
    
    Functions = [Tanh, Logistic, Identity]
    dFunctions = [dTanh, dLogistic, dIdentity]
    ''' Activation Functions '''
    
    N = Layers[0] # No. input neurons
    M = Layers[-1] # No. output neurons
    L = len(Layers) # No. layers
    
    dAF = [dFunctions[f] for f in ActivationFunctions] # derivation of Activation function calculated using the original function
    ActivationFunctions = [Functions[f] for f in ActivationFunctions] # activation functions could be chosen from the three pre-defined using their indecis (0: Tanh, 1:Logistic, 2:Identity)

            
    # three types of initializing the weights of NN: 1. by passing it to function in the following form:
    ''' an array consist of (No. Layers - 1) arrays, for each hidden layer and the output layer,
     in which there are (No. Neurons in the corresponding layer) arrays, for each neuron in the layer,
     consist of (No. Neurons within the previous layer + 1) real numbers for weights of each input to the neuron and BIAS '''
    if not Weights: # 2. randomly initializing it using an arbitrary seed (Weights = '' or [])
        Weights = [[[(rand.random() * 4 - 2) for k in range(Layers[l - 1] + 1)] for n in range(Layers[l])] for l in range(1, L)]
    elif type(Weights) == str: # 3. from a file in which weights has been saved in the above-mentioned structure
        data = readNums(File)
        Weights = [[[0 for k in range(Layers[l - 1] + 1)] for n in range(Layers[l])] for l in range(1, L)]
        d = 0
        for l in range(1, L):
            for n in range(Layers[l]):
                for k in range(Layers[l - 1] + 1):
                    Weights[l][n][k] = data[d]
                    d = d + 1
    # two types of passing training patterns, 1. direct input 2. from a file, it's similar to weights but in a structure as explained within readPatterns function
    if type(Patterns) == str:
        Patterns = readPatterns(Patterns, N, M)
    def MLP(X, W, AF): # the MLP as a function which passes input(X) through the network with calculated weights(W) and Activation functions(AF) and returns outputs of every layers
        YL = X
        Y = []
        for l in range(len(W)):
            Y = Y + [[1] + YL]
            YL = MPNeuron(Y[l], W[l], AF[l])
        return Y + [YL]
    ''' MLP Initialization ''' #BP-Phase 1
    ### Part 2 ###
    
    def addWeights(W1, W2): #pairwise sum of two lists
        if not W1:
            return W2
        elif not W2:
            return W1
        elif type(W1) == list:
            return [addWeights(a, b) for a, b in zip(W1, W2)]
        else:
            return W1 + W2
        
    def DeltaRule(DeltaK, Outs, W, dAFs, Zeta): # Recursively computes Delta and weight changes for each layer
        dw = [[Zeta[-1] * delta * out for out in Outs[-1]] for delta in DeltaK]
        if not dAFs:
            return [dw]
        f = dAFs[-1]
        DeltaH = [sum([DeltaK[k] * W[-1][k][h] for k in range(len(DeltaK))]) * f(Outs[-1][h]) for h in range(1, len(Outs[-1]))]
        return DeltaRule(DeltaH, Outs[:-1], W[:-1], dAFs[:-1], Zeta[:-1]) + [dw]

    Err = []
    for s in range(MaxSteps): # in each step we will shuffle all the training patterns and use every one of them for learning
        dW = []
        rand.shuffle(Patterns)
        for p in Patterns:
            X = p[0]
            Yt = p[1]
            Output = MLP(X, Weights, ActivationFunctions)
            Y = Output[-1]
            Err = Err + [sum([(a - b)**2 for a, b in zip(Yt, Y)])/2]
            DeltaM = [(Yt[m] - Y[m]) * dAF[-1](Y[m]) for m in range(M)] # Computing Delta for output layer
            dW = DeltaRule(DeltaM, Output[:-1], Weights, dAF[:-1], LearningRates)
            newWeights = addWeights(Weights, dW)
            if not Batch: # in single step mode (Batch = false) we'll apply changes for each patterns
                Weights = newWeights
        if Batch: # in batch mode changes will apply after using the whole dataset of training patterns
            Weights = newWeights
    return Err, lambda X: MLP(X, Weights, ActivationFunctions)[-1]

def gnuplotOut(Dir, X, Y, xlabel, ylabel, title):
    file = open(Dir, 'w')
    file.write('set title'+ '"' + title + '"\n')
    file.write('set xlabel'+ '"' + xlabel + '"\n')
    file.write('set ylabel'+ '"' + ylabel + '"\n')
    file.write('plot [' + str(min(X)) + ':' + str(max(X)) + '] [' + str(min(Y)) + ':' + str(max(Y)) + '] '"'-'"' with line\n')
    file.write('# x\ty\n')
    for x, y in zip(X, Y):
        file.write(str(x) + ' ' + str(y) + '\n')
    file.write ('e')
    file.close ()
