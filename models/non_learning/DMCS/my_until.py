DEBUG = 0  # Set to 1 for debugging mode

class Edge:
    def __init__(self, u, v):
        if u > v:
            u, v = v, u
        self.u = u
        self.v = v

    def __lt__(self, other):
        if isinstance(other, Edge):
            return (self.u, self.v) < (other.u, other.v)
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.u == other.u and self.v == other.v
        return NotImplemented

    def __hash__(self):
        return hash(self.u) ^ (hash(self.v) << 1)

def make_edge(u, v):
    return Edge(u, v)
