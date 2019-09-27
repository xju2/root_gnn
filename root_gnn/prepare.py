"""
split one graph into input graph and output graph
"""
import numpy as np

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    if dphi > np.pi:
        dphi -= 2*np.pi
    if dphi < -np.pi:
        dphi += 2*np.pi
    return dphi

def graph_to_input_target(graph):
    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos",)
    input_edge_fields = ("distance",)

    input_global_fields = ('attributes',)
    target_global_fields = ('solution',)

    input_graph = graph.copy()
    target_graph = graph.copy()

    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields)
        )
        target_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields)
        )

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields)
        )
        target_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields)
        )

    input_graph.graph['features'] = create_feature(input_graph.graph, input_global_fields)
    target_graph.graph['features'] = create_feature(target_graph.graph, target_global_fields)

    return input_graph, target_graph


