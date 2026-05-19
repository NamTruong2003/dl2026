from Node import Node
class Layer:
    def __init__(self):
        self.list_node = []
        self.weight = []
        self.bias = []

    def add_nodes(self,nodes):
        for node in nodes:
            self.list_node.append(node) 
    def add_weights(self,weights):
        self.weight=weights
    def add_bias(self,bias):
        self.bias = bias
    def get_bias(self):
        return self.bias
    def get_weights(self):
        return self.weight
    def get_data(self):
        return [node.get_data() for node in self.list_node]
    def update_data(self,data):
        if len(data) != len(self.list_node):
            print(f"data is different size {len(data)} from list node {len(self.list_node)}")
        else:
            for i in range(len(data)):
                self.list_node[i].update_data(data[i])
    @staticmethod
    def create_layer(list_of_layer)->List[Layer]:
        list = []
        
        for i in range(len(list_of_layer)):
            nodes = Node.create_node(int(list_of_layer[i]))
            layer = Layer()
            layer.add_nodes(nodes)
            if i != len(list_of_layer)-1:
                weights = [[0 for _ in range(int(list_of_layer[i+1]))] for _ in range(int(list_of_layer[i]))]
                bias = [0] * int(list_of_layer[i+1])
                layer.add_weights(weights)
                layer.add_bias(bias)
            list.append(layer)
            print(f"Add layer {i} successfully ")
        print(f"Create layers successfully with lenght {len(list)}")
        return list
    
        
            
