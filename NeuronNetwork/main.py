from utils import *
from NeuronNetwork import NeuronNetwork

if __name__ == "__main__":
    
    #n_layer,n_neuron = readInput("input.txt")
    #print(n_layer)
    #print(n_neuron)
    neuronNetwork = NeuronNetwork()
    neuronNetwork.create_from_config("input_with_weights.txt")
    neuronNetwork.print_weights()
    X = [[0,0],[1,0],[0,1],[1,1]]
    Y = [0,1,1,0]
    neuronNetwork.train(X,Y,80000,0.5)
    neuronNetwork.print_weights()
    
    print("\n=== XOR TESTING RESULTS ===")
    X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for i in range(len(X_test)):
        inputs = X_test[i]
        neuronNetwork.activate(inputs)
        predict = neuronNetwork.list_layer[-1].get_data()
        
        if isinstance(predict, list):
            val = predict[0][0] if isinstance(predict[0], list) else predict[0]
        else:
            val = predict
            
        print(f"Input: {inputs} -> Sigmoid Probability: {val:.6f} -> Output: {round(val)}")



    