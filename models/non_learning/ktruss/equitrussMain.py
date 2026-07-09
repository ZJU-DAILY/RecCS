import os
import time
from MyGraph import MyGraph,MyEdge
from TecIndexSB import TecIndexSB
from TecIndexG import TecIndexG
from SGN import SGN

class MainF:
    def __init__(self):
        print("Welcome to Equitruss")
    @staticmethod
    def main(args):
        # Initialize parameters
        if len(args) == 3:
            c = int(args[0])
            file_name = args[1]
            pathtec = args[2]
        else:
            c = 1
            print("Path for graph file and path for index files is not given. Toy graph will be used")
            file_name = "data/toyg.txt"
            pathtec = "toy"

        tec = TecIndexSB()
        print(file_name)
        mg = MyGraph()

        # Read Graph
        print("start")
        start_time = time.time()
        mg.read_graph_edgelist(file_name)
        end_time = time.time()
        print(f"Graph read time: {end_time - start_time:.2f} seconds")
        
        if c == 1:
            klistdict = {}
            trussd = {}

            # Create directory for index files
            MainF.create_dir(pathtec)

            # Truss Decomposition
            start_time = time.time()
            klistdict = mg.compute_truss(pathtec, trussd)
            end_time = time.time()
            print(f"Truss computation time: {end_time - start_time:.2f} seconds")
            mg.write_support(f"{pathtec}/truss.txt", trussd)

            # Create Index
            start_time = time.time()
            tec.construct_index(klistdict, trussd, mg)
            end_time = time.time()
            print(f"Index for given graph is created. Index creation time: {end_time - start_time:.2f} seconds")
            tec.write_index(pathtec)
        else:
            if not os.path.exists(pathtec):
                print("Given path for index files does not exist")
                exit(0)
            tec.read_index(mg, pathtec)
            print("Index files are read and index is created")
            MainF.test(tec)

    @staticmethod
    def test(tec):
        print("Enter query node id and k truss value")
        query = int(input())
        while query != -1:
            k = int(input())
            start_time = time.time()
            com = tec.find_k_community_for_query(query, k)
            end_time = time.time()

            if not com:
                print("There is no community for this query with given truss value")
            else:
                for community in com:
                    for edge in community:
                        print(f"({edge.s}, {edge.t}), ", end="")
                    print()

            print(f"Query time: {end_time - start_time:.2f} seconds")
            print("\n\nEnter query node id and k truss value, -1 to exit")
            query = int(input())

    @staticmethod
    def create_dir(path):
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        except Exception as e:
            print("There is a problem with creating directory:", e)

# Entry point for the program
if __name__ == "__main__":
    import sys
    MainF.main(sys.argv[1:])
