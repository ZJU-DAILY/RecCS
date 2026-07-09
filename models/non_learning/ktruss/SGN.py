class SGN:
    def __init__(self, truss, tnid):
        self.truss = truss  # Value of the truss
        self.idd = tnid      # Node ID
        self.edgelist = []   # List to hold edges

    def add_edge(self, e):
        self.edgelist.append(e)  # Method to add an edge to the edge list
