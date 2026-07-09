import sys
from dmcs import DMCS
from my_graph import MyGraph
DEBUG = 0

def read_query_nodes(file_path: str) -> list:
    try:
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: cannot open file {file_path}")
        return []

def main():

    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} graph_file_path query_file_path output_file_path")
        return 1
    
    s_graph_path = sys.argv[1]
    s_query_path = sys.argv[2]
    s_output_path = sys.argv[3]

   
    my_graph = MyGraph()
    my_graph.read_graph_from_file(s_graph_path)
    # if DEBUG:
    #     my_graph.print_graph()
 
    query_nodes = read_query_nodes(s_query_path)
    if not query_nodes:
        return 1

  
    dmcs = DMCS(my_graph)
    dmcs.get_a_community_by_fpa(query_nodes, s_output_path)
    dmcs.print_returned_community(s_output_path)

if __name__ == "__main__":
    main()