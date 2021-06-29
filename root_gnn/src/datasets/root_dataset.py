import numpy as np
import itertools

from graph_nets import utils_tf
from graph_nets import utils_np
from graph_nets import graphs
from root_gnn.src.datasets.base import DataSet
import ROOT
from functools import partial 
from typing import Optional  

def make_graph(feature_values_with_truth, debug: Optional[bool] = False):#, debug=False):
    """Creates a GraphsTuple for an event
    
    Parameters:
    feature_values_with_truth = contains the feature values for all the particles (node attributes) and global truth
    
    Returns:
    Input_graph = Input GraphsTuple
    Target_graph = Target GraphsTuple
    """
    
    #print("PROCESSING GRAPH") 
    truth = feature_values_with_truth[1]
    feature_values = feature_values_with_truth[0] 
    n_nodes = len(feature_values) 
    all_edges = list(itertools.combinations(range(n_nodes), 2))
    senders = np.array([x[0] for x in all_edges])
    receivers = np.array([x[1] for x in all_edges])
    n_edges = len(all_edges)
    edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
    zeros = np.array([0.0], dtype=np.float32)
    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": feature_values,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([n_nodes], dtype=np.float32)
    }
    target_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": feature_values,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": truth 
    }
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
    return [(input_graph, target_graph)]

def branch_names_to_feature_values(event, branches, filterings, include_particle_type=False):  #default value for truth is float? 
    """For a particular event, outputs the features for all particles as well as the global truth. 
    Parameters:
    event = current event (TTree row)
    branches = dictionary of {particle-type-name : particle-type's branch names}
    filterings = list of particle filtering queries to process
    include_particle_type = True if particle type is included as a feature, False otherwise 
    
    Returns:
    feature_values = list of all the features for each particle 
    truth = the global truth (what you are trying to predict)
    """
    
    truth_attribute="TRUTH"
    filter_attribute="FILTER"
    def filter_func(event, particle_name, index):
        if filterings.get(particle_name) is None:
            return True
        for filtering in filterings[particle_name]:
            if not eval(filtering):
                return False
        return True 
    
    feature_values = []
    particle_mapping = {v:k for k,v in enumerate(branches.keys())}
    for particle_name, particle_branches in branches.items():
        if particle_name == filter_attribute:
            continue
        if particle_name == truth_attribute: #truth value
            if len(particle_branches) > 1 or not isinstance(particle_branches[0], str): #user did not pass in a branch name to be the truth
                truth = particle_branches
            else: #user passed in a single branch name
                truth_branch = getattr(event, particle_branches[0])
                if not isinstance(truth_branch, (int, float, complex)): #is a list
                    truth = [elem for elem in truth_branch]
                else:
                    truth = truth_branch 
            continue
        else:
            num_nodes = len(getattr(event, particle_branches[0])) 
            particle_features = [[getattr(event, branch_name)[index] for branch_name in branches[particle_name] if filter_func(event, particle_name, index)] 
                                     for index in range(num_nodes)]
            print(particle_name, particle_features)
            mapped_value = particle_mapping[particle_name]
            particle_features = [feature + ([mapped_value] if include_particle_type else []) for feature in particle_features] 
            feature_values += particle_features
    return (feature_values, truth)

def yaml_to_branch_names(tree, yaml_file):
    """Reads the configuration yaml_file that specifies branches to read from. 
    Parameters:
    tree = TTree to read 
    yaml_file = name of yaml file to read  
    
    Returns:
    branches = dictionary of {particle-type-name : particle-type's branch names}
    filters = list of filtering queries to be processed
    """
    import yaml
    from collections import OrderedDict
    branches = OrderedDict() 
    truth_attribute = "TRUTH"
    ordering_attribute = "NODE_ORDERS"
    filter_attribute = "FILTER"
    foundTruth = False
    with open(yaml_file, "r") as file:
        documents = yaml.full_load(file)    
        order = documents.get(ordering_attribute)
        if order is None:
            print("ERROR: NODE ORDER NOT SPECIFIED")
            return 
        
        for particle in order:
            if documents[particle] is None:
                print("ERROR: Particle specified in NODE_ORDERS does not exist in yaml_file")
                return None, None
            branches[particle] = documents[particle]
            for branch_name in documents[particle]:
                try:
                    branch = getattr(tree, branch_name) 
                except: 
                    print("ERROR: BRANCH NAME", branch_name, "DOES NOT EXIST")
                    return  

    truth = documents.get(truth_attribute)
    if truth is None:
        print("ERROR: TRUTH NOT SPECIFIED")
        return None 
    if not check_valid_truth(tree, truth):
        print("ERROR: TRUTH INPUT HAS INCONSISTENT VALUE")
        return None
    branches[truth_attribute] = truth
        
    #to filter preprocess filter requests into a boolean string that eval() can evaluate for each particle in an event
    filter_list = documents.get(filter_attribute)
    filters = OrderedDict() 
    special_filter_chars = '==>=<=+-**/'
    if filter_list is not None:
        for particle_filter in filter_list: 
            for particle, filterings in particle_filter.items():
                filters[particle] = []
                for filtering in filterings:
                    #Split the filter request by blank space. For each word that's a branch name, replace it with getattr(...) - this is a form that can be easily evaluated by eval() 
                    entry_list = filtering.split() 
                    temp = [text if (text.isdigit() or (text in special_filter_chars)) else "getattr(event,'" + text + "')[index]" for text in entry_list]
                    filters[particle].append(" ".join(temp))
    return branches, filters

def check_valid_truth(event, truth_branch_name, max_events = 1000): 
    """Checks if an inputted truth attribute is valid if the truth attribute is a branch name. 
    If the branch elements are a list, checks that all branch elements have the same shape. 
    If the branch elements are a number, checks that all branch elements are the same value. 
    Parameters:
    event = TTree to read from
    truth_attribute = key name in the yaml file that corresponds to the truth 
    max_events = maximum number of events to iterate through for truth validation
    
    Returns:
    True if truth attribute is valid
    False if truth attribute is not valid
    """
    #If truth_attribute is a branch name, check if branch is a list OR a single value. 
    #If it's a single number then it needs to have same value for all events! 
    if len(truth_branch_name) == 1 and isinstance(truth_branch_name[0], str): #truth_attribute is a branch name
        truth_value = getattr(event, truth_branch_name[0]) 
        num_events = min(max_events, event.GetEntries())
        event.GetEntry(0)
        if not isinstance(truth_value, (int, float, complex)): #check all lists have same shape
            size = len(truth_value)
            for i in range(1, num_events):
                event.GetEntry(i)
                if len(truth_value) != size:
                    return False 
            return True 
        initial_entry = truth_value #otherwise... check all numbers have same value 
        for i in range(1, num_events):  
            event.GetEntry(i)
            if truth_value != initial_entry:
                return False 
        return True 
    return True 

def read(filename, yaml_filename): #add tree_name argument here? 
    print("ROOT DATASET READ TRIGGERED")
    rfile = ROOT.TFile.Open(filename, 'READ')
    tree = rfile.Get('nominal_Loose')
    n_entries = tree.GetEntries() 
    tree.GetEntry(0) 
    branches, filters = yaml_to_branch_names(tree, yaml_filename)  
    #Check the truth here before the for loop (run it for 1000 events max) 
    for ientry in range(n_entries):
        tree.GetEntry(ientry)
        temp = branch_names_to_feature_values(tree, branches, filters)
        #print(temp)
        yield temp

class RootDataset(DataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       # self.read = read
        self.make_graph = make_graph

    def set_config_file(self, yaml_filename):
        #self.yaml_file = yaml_filename 
        self.read = partial(read, yaml_filename=yaml_filename)

    def set_include_particle_type(self, include_particle_type):
        self.branch_names_to_feature_values = partial(branch_names_to_feature_values, include_particle_type=include_particle_type)
    
    def _num_evts(self, filename):
        rfile = ROOT.TFile.Open(filename, 'READ')
        tree = rfile.Get('nominal_Loose')
        num_entries = tree.GetEntries() 
        rfile.Close()
        return num_entries 