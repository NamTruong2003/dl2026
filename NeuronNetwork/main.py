from utils import *
from NeuronNetwork import NeuronNetwork
if __name__ == "__main__":
    input = [1,1]
    #n_layer,n_neuron = readInput("input.txt")
    #print(n_layer)
    #print(n_neuron)
    neuronNetwork = NeuronNetwork()
    neuronNetwork.create_from_config("input_with_weights.txt")
    neuronNetwork.print_weights()
    neuronNetwork.activate(input)
    
    neuronNetwork.print_data()
    


    