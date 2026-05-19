from Layer import Layer
from utils import *
class NeuronNetwork:
    def __init__(self):
        self.list_layer = []
    def add_layer(self,layer):
        self.list_layer = layer
    def print_weights(self):
        print("Weight of Network")
        for layer in self.list_layer:
            print(layer.get_weights())
            print(layer.get_bias())
    def create_layers(self,l_layers):
        self.list_layer = Layer.create_layer(l_layers)
    def activate(self,input):
        self.list_layer[0].update_data(input)
        for i in range(len(self.list_layer)-1):
            z_list = feedForward(self.list_layer[i].get_data(),self.list_layer[i].get_weights(),self.list_layer[i].get_bias())
            if isinstance(z_list, str):
                print(z_list) 
                exit()  
            data_new = active(z_list)
            self.list_layer[i+1].update_data(data_new)
        print("Activate success fully")
    def print_data(self):
        print("Data of Network")
        for layer in self.list_layer:
            print(layer.get_data())
    def create_from_config(self, config):
        network_data = parse_nn_config(config)
        if network_data:
            layers = network_data['num_layers']
            topology = network_data['neurons_per_layer']
            
            print(f"Total Layers: {layers}")
            print(f"Network Topology: {topology}\n")
            self.create_layers(topology)
            
            for layer_name, data in network_data['layers_data'].items():
                layer_biases = data['biases']
                layer_weights = data['weights']
                
                layer_idx = int(layer_name.split('_')[1])
                
                self.list_layer[layer_idx - 1].add_weights(layer_weights)
                self.list_layer[layer_idx - 1].add_bias(layer_biases)
                
                print(f"--- {layer_name} ---")
                print(f"Biases: {layer_biases}")
                print(f"Weights matrix: {layer_weights}")
                print()

            