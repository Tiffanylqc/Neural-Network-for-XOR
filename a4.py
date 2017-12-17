import random
import math
import time
import copy

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 6
#####################################################
#####################################################


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))

def logistic_derivative(x):
    return logistic(x) * (1-logistic(x))

class Neuron:
    def __init__(self, attribute_weights, neuron_weights, bias_weight):
        # neuron.attribute_weights[i] = Weight of input attribute i as input to this neuron
        self.attribute_weights = attribute_weights
        # neuron.neuron_weights[i] = Weight of neuron j as input to this neuron
        self.neuron_weights = neuron_weights
        self.bias_weight = bias_weight

class ANN:
    def __init__(self, num_attributes, neurons):
        # Number of input attributes.
        self.num_attributes = num_attributes
        # Number of neurons. neurons[-1] is the output neuron.
        self.neurons = neurons
        for neuron_index, neuron in enumerate(self.neurons):
            for input_neuron, input_weight in neuron.neuron_weights.items():
                assert(input_neuron < neuron_index)
        # initialize the a_values, in_values, deltas
        self.in_values=[]
        self.a_values=[]

    # Calculates the output of the output neuron for given input attributes.
    # propage the attributes(input) to calculate the outputs
    # record every <in> value for each neuron
    def calculate(self, attributes):
        ###########################################
        # Start your code
        self.in_values=[]
        self.a_values=[]

        # calculate the input value from attribute to the hidden neurons
        for neuron_index in range(len(self.neurons)-1):
            in_value=0
            neuron=self.neurons[neuron_index]
            # add the bias weight
            in_value+=neuron.bias_weight*(-1)
            for attri_index in range(len(attributes)):
                in_value+=attributes[attri_index]*neuron.attribute_weights[attri_index]
            self.in_values.append(in_value)
            a_value=logistic(in_value)
            self.a_values.append(a_value)

        # calculate the input value from other neurons to this neuron
        in_value=0
        # add bias weight
        in_value+=(self.neurons[-1].bias_weight)*(-1)
        for hid_neuron_idx in range(len(self.neurons)-1):
            in_value+=(self.a_values[hid_neuron_idx])*(self.neurons[-1].neuron_weights[hid_neuron_idx])
        self.in_values.append(in_value)
        a_value=logistic(in_value)
        self.a_values.append(a_value)

        # return the output of the output neuron
        return self.a_values[-1]
        # End your code
        ###########################################

    # Returns the squared error of a collection of examples:
    # Error = 0.5 * sum_i ( example_labels[i] - ann.calculate(example_attributes) )**2
    def squared_error(self, example_attributes, example_labels):
        ###########################################
        # Start your code
        error=0
        for example_idx,example in enumerate(example_attributes):
            output=self.calculate(example)
            label=example_labels[example_idx]
            error+=(output-label)**2
        return error*0.5

        # End your code
        ###########################################

    # Runs backpropagation on a single example in order to
    # update the network weights appropriately.
    def backpropagate_example(self, attributes, label, learning_rate=1.0):
        ###########################################
        # Start your code

        # do forward calculation from input attributes to output neuron
        # record the <in> values and <a> values for each neuron
        self.calculate(attributes)

        # propage deltas backward from output layer to input layer
        deltas=[0 for neuron in self.neurons] # initialize deltas with all zeros

        # calculate delta for the output neuron
        deltas[-1]=(label-self.a_values[-1])*logistic_derivative(self.in_values[-1])

        # calculate deltas for hidden neurons
        for idx in range(len(self.neurons)-1):
            deltas[idx]=logistic_derivative(self.in_values[idx])\
                        *self.neurons[-1].neuron_weights[idx]*deltas[-1]

        # update the weights
        self.neurons[-1].bias_weight += learning_rate * (-1) * deltas[-1]
        for neuron_idx in range(len(self.neurons) - 1):
            self.neurons[-1].neuron_weights[neuron_idx] += learning_rate * self.a_values[neuron_idx] * deltas[-1]
            self.neurons[neuron_idx].bias_weight += learning_rate * (-1) * deltas[neuron_idx]
            for i, attribute in enumerate(attributes):
                self.neurons[neuron_idx].attribute_weights[i] += learning_rate * attribute * deltas[neuron_idx]

        # End your code
        ###########################################

    # Runs backpropagation on each example, repeating this process
    # num_epochs times.
    def learn(self, example_attributes, example_labels, learning_rate=1.0, num_epochs=100):
        ###########################################
        # Start your code

        for _ in range(num_epochs):
            for idx,example in enumerate(example_attributes):
                self.backpropagate_example(example, example_labels[idx], learning_rate)

        # End your code
        ###########################################


example_attributes = [ [0,0], [0,1], [1,0], [1,1] ]
example_labels = [0,1,1,0]

def random_ann(num_attributes=2, num_hidden=2):
    neurons = []
    # hidden neurons
    for i in range(num_hidden):
        attribute_weights = {attribute_index: random.uniform(-1.0,1.0) for attribute_index in range(num_attributes)}
        bias_weight = random.uniform(-1.0,1.0)
        neurons.append(Neuron(attribute_weights,{},bias_weight))
    # output neuron
    neuron_weights = {input_neuron: random.uniform(-1.0,1.0) for input_neuron in range(num_hidden)}
    bias_weight = random.uniform(-1.0,1.0)
    neurons.append(Neuron({},neuron_weights,bias_weight))
    ann = ANN(num_attributes, neurons)
    return ann

best_ann = None
best_error = float("inf")
for instance_index in range(10):
    ann = random_ann()
    ann.learn(example_attributes, example_labels, learning_rate=10.0, num_epochs=10000)
    error = ann.squared_error(example_attributes, example_labels)
    if error < best_error:
        best_error=error
        best_ann = ann

print(best_ann.neurons[0].attribute_weights[0])
print(best_ann.neurons[0].attribute_weights[1])
print(best_ann.neurons[0].bias_weight)
print(best_ann.neurons[1].attribute_weights[0])
print(best_ann.neurons[1].attribute_weights[1])
print(best_ann.neurons[1].bias_weight)
print(best_ann.neurons[2].neuron_weights[0])
print(best_ann.neurons[2].neuron_weights[1])
print(best_ann.neurons[2].bias_weight)
print("\n")
print(best_error)

#####################################################
#####################################################
# # Please hard-code your learned ANN here:
# learned_ann = random_two_layer_ann()
# learned_ann.neurons[0].attribute_weights[0] = -7.411382835861137
# learned_ann.neurons[0].attribute_weights[1] = -7.649980751250681
# learned_ann.neurons[0].bias_weight = -3.2331877341284003
# learned_ann.neurons[1].attribute_weights[0] = -6.082932958761941
# learned_ann.neurons[1].attribute_weights[1] = -6.127386006088293
# learned_ann.neurons[1].bias_weight = -9.11911243973505
# learned_ann.neurons[2].neuron_weights[0] = -12.650118542123902
# learned_ann.neurons[2].neuron_weights[1] = 12.486082982430746
# learned_ann.neurons[2].bias_weight = 6.035172356749271
# # Enter the squared error of this network here:
# final_squared_error = 2.4965988804590195e-05
#####################################################
#####################################################


