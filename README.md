# Multi-layer Perzeptron

[Multi-layer Perzeptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron) (Perceptron) using [Backpropagation of Error](https://en.wikipedia.org/wiki/Backpropagation) to train the model. 
Implemented in python, for a programming assignment of course ***Technical Neuroal Network***.
 
# Using Instruction

Simply import function "MultiLayerPerzeptronBP" from MultiLayerPerzeptronBP.py

	$ python
	from MultiLayerPerzeptronBP import MultiLayerPerzeptronBP
	
Now you can use this function with the following definition:

	Errors, TNN = MultiLayerPerzeptronBP (Layers, ActivationFunctions, Weights, Patterns, LearningRates, RandomSeed, MaxSteps, Batch)

### Layers

A list containing the number of neurons in each layer, e.g., for N-H-M it’s [N, H, M].

### ActivationFunctions

A list containing an integer for each layer (except the input layer) which is corresponding to a predefined activation function 0: Tanh, 1: Logistic, 2: Identity

### Weights(including BIAS): Initial weights

A list consist of (No. Layers - 1) arrays, for each hidden layer and the output layer, in which there are (No. Neurons in the corresponding layer) arrays, for each neuron in the layer, 
consist of (No. Neurons within the previous layer + 1) real numbers for weights of each input to the neuron and BIAS, 
**Alternatively**, path of a .dat file in which you must put weights with the aforementioned order (lines with # consider as a comment), 
**Alternatively**, '' or [] for random initialization with a value between -2 and 2

### Patterns

A list of P patterns used to train the perzeptron in the form of tuples in which the first element is a list containing N inputs and the second one is a list containing M outputs.
**Alternatively**, source direction of  a .dat file in which for each training pattern you must put the inputs(without BIAS) followed by outputs followed by next patterns (lines with # consider as a comment).

### LearningRate

A list containing learning rates for each layer except input layer(all the neurons of a layer share the same learning rate and activation function).

### RandomSeed

A random seed used for random initializing and shuffling, to be able to reproduce results.

### MaxSteps

The maximum number of iterations in which the model can train.

### Batch

A boolean value which is 1 for batch learning and 0 for single-step learning

### Errors

A list containing the squared error of each pattern in each of the iterations.
***To plot the learning curve of the model using gnuplot, a function called gnuplotOut which makes a file readable for gnuplot had been implemented. Check the examples for more details.***

![See XOR_learning_curve.png as sample learning curve for XOR MLP model plotted by gnuplot](/XOR_learning_curve.png)

### TNN

A function which is a MLP model with calculated weights, with the following definition:

	Y = TNN(X)
	
**X**: a list of containing N values as input
**Y**: a list containing M values as output

# Example:

*Please check "MultiLayerPerzeptronBP-Test.py" for some examples.*

# Authors 

	Ali Mohammadi
	Rozhin Bayati


*Best Regards*