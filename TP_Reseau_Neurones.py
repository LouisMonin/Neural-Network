from random import random
import math



#Créer un RNA avec un couche cachée

def initializeNetwork(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs)], 'bias':random()} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden)], 'bias':random()} for i in range(n_outputs)]
	network.append(output_layer)
	return network

#Montrer l'architecture du réseau
    
def showNetwork(nn):
    hidden_layer = nn[0]
    n_hidden = len(hidden_layer)
    first_neuron_weights = hidden_layer[0]['weights']
    n_inputs = len(first_neuron_weights)
    output_layer = nn[1]
    n_outputs = len(output_layer)
    print("Nombre de neurones dans la couche cachée:",n_hidden)
    print("Nombre de neurones dans la couche d'entrée: ",n_inputs)
    print("Nombre de neurones dans la couche de sortie:",n_outputs)
	
#Calcul de la sortie du réseau
def neuronOutput(neuron, inputs):
    weights = neuron['weights']
    bias = neuron['bias']

    combination = bias
    for i in range(len(weights)):
      combination += weights[i] * inputs[i]
    output = 1.0/(1.0+math.exp(-combination))
    return output

#Calculs effectués par le RNA de l'entrée jusqu'à la sortie
def forward_propagate(network, inputs):
    new_inputs = inputs
    nbLayers = len(network)
    for numl in range(nbLayers):
        layer = network[numl]
        nbNeurons_Layer = len(layer)
        outputs = []
        for numn in range(nbNeurons_Layer):
          neuron = layer[numn]
          neuron['output'] = neuronOutput(neuron, new_inputs)
          outputs.append(neuron['output'])
        if (numl<nbLayers-1):
          new_inputs = outputs
    return outputs
	
	
#Script principal


nn = initializeNetwork(4, 5, 3)
showNetwork(nn)

inputs = [1.0, 1.0, 1.0, 1.0]

outputs = forward_propagate(nn, inputs)
print("Entrées du réseau:",inputs)
print("Sorties du réseau:",outputs)