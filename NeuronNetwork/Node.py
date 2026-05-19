class Node:
    def __init__(self,data:Integer):
        self.data = data
    def update_data(self,data):
        self.data = data
    def get_data(self):
        return self.data
    @staticmethod
    def create_node(number_of_nodes)->List[Node]:
        list = []
        for i in range(number_of_nodes):
            list.append(Node(0))
        return list