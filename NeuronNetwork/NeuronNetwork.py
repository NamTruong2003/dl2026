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

    def back_propagation(self, Y):
        if not isinstance(Y, list):
            Y = [Y]
        gradient = []
        predict = self.list_layer[-1].get_data()
        
        is_predict_2d = isinstance(predict, list) and len(predict) > 0 and isinstance(predict[0], list)
        minus_predict = [[-v for v in row] for row in predict] if is_predict_2d else [-val for val in predict]
        
        error_output = matrixAddition(Y, minus_predict)
        gradient_output = elementwise_multiply(error_output, sigmoid_derivative(predict))
        
        len_layer = len(self.list_layer)
        for i in range(len_layer - 1, 0, -1):
            if i == len_layer - 1:
                gradient.append(gradient_output)
            else:
                gradient_next_layer = gradient[-1]
                weights_current_layer = self.list_layer[i].get_weights()
                
                error = matrixMultiply(weights_current_layer, gradient_next_layer)
                current_layer_data = self.list_layer[i].get_data()
                gradient_layer = elementwise_multiply(error, sigmoid_derivative(current_layer_data))
                gradient.append(gradient_layer)
                
        gradient.reverse()
        return gradient

    def update_weight(self, gradient, learning_rate=0.5):
        len_layer = len(self.list_layer)
        
        for i in range(0, len_layer - 1):
            layer = self.list_layer[i]
            
            grad = gradient[i]
            inputs = layer.get_data()
            
            is_grad_1d = not isinstance(grad, list) or (len(grad) > 0 and not isinstance(grad[0], list))
            is_in_1d = not isinstance(inputs, list) or (len(inputs) > 0 and not isinstance(inputs[0], list))
            
            weights = layer.get_weights()
            biases = layer.get_bias()
            
            if is_grad_1d:
                for h in range(len(biases)):
                    biases[h] += learning_rate * grad[h]
                    
                for r in range(len(weights)):
                    for h in range(len(weights[0])):
                        input_val = inputs[r] if is_in_1d else inputs[r]
                        weights[r][h] += learning_rate * grad[h] * input_val
            else:
                for row in range(len(grad)):
                    for h in range(len(biases)):
                        biases[h] += learning_rate * grad[row][h]
                        
                    for r in range(len(weights)):
                        for h in range(len(weights[0])):
                            input_val = inputs[row][r] if not is_in_1d else inputs[r]
                            weights[r][h] += learning_rate * grad[row][h] * input_val

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                inputs = X[i]
                target = Y[i]
                
                
                
                self.activate(inputs)
                    
                predict = self.list_layer[-1].get_data()
                
                is_target_list = isinstance(target, list)
                is_predict_list = isinstance(predict, list)
                
                if not is_target_list and not is_predict_list:
                    total_loss += 0.5 * (target - predict) ** 2
                elif is_target_list and is_predict_list:
                    for idx in range(len(target)):
                        total_loss += 0.5 * (target[idx] - predict[idx]) ** 2
                        
                gradient = self.back_propagation(target)
                self.update_weight(gradient, learning_rate)
                
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(X):.6f}")

            



        