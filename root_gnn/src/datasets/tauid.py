import numpy as np
import math
import itertools

from graph_nets import utils_tf
from root_gnn.utils import calc_dphi
from root_gnn.src.datasets.base import DataSet

import time
import os
from multiprocessing import Pool
from functools import partial
import tensorflow as tf
from root_gnn.src.datasets import graph
from root_gnn.utils import load_yaml
from sklearn.neighbors import NearestNeighbors



tree_name = "output"

def pad_array(arr,size,n_attrs):
        arr.extend([[0.]*n_attrs]*(size-len(arr)))

def make_graph(chain, debug=False, connectivity=None, 
               signal=None, with_edge_features=False, with_node_type=True, 
               with_hlv_features=False, use_delta_angles=True,
               tower_lim=None, track_lim=None, cutoff=False,
               background_dropoff=0., rand=None,
               use_jetPt=True, use_jetVar=True):
    isTau = 0
    track_idx = 0
    tower_idx = 0
    graph_list = []

    
    isTauEvent = chain.nTruthJets > 0
    for ijet in range(chain.nJets):
                   
        # Match jet to truth jet that minimizes angular distance
        split_point = 0
        nodes = []
        tower_nodes = []
        track_nodes = []
        min_index = 0
        if chain.nTruthJets > 0:
            min_dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[0])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[0])**2)
        for itruth in range(chain.nTruthJets):
            dR = math.sqrt(calc_dphi(chain.JetPhi[ijet],chain.TruthJetPhi[itruth])**2 + (chain.JetEta[ijet]-chain.TruthJetEta[itruth])**2)
            if dR < min_dR:
                min_dR = dR
                min_index = itruth
        if chain.nTruthJets > 0 and min_dR < 0.4:
            isTau = chain.TruthJetIsTautagged[min_index]
        else:
            isTau = 0
        isTau = 0 if (isTau != 1 and isTau != 3) else isTau
        

        ### Nodes ###
        len_tower_attr = 7
        for itower in range(chain.JetTowerN[ijet]):
            if(not cutoff or chain.JetTowerEt[tower_idx] >= 1.0):
                
                if use_delta_angles:
                    deta = chain.JetEta[ijet]-chain.JetTowerEta[tower_idx]
                    dphi = calc_dphi(chain.JetPhi[ijet],chain.JetTowerPhi[tower_idx])
                    tower_feature = [0.0, math.log10(chain.JetPt[ijet]),
                                     math.log10(chain.JetTowerEt[tower_idx]),
                                     math.fabs(deta),
                                     math.fabs(dphi),
                                     0.0,
                                     0.0]
                else:
                    tower_feature = [0.0, math.log10(chain.JetPt[ijet]),\
                                     math.log10(chain.JetTowerEt[tower_idx]),\
                                     chain.TowerEta[tower_idx]/3,\
                                     chain.TowerPhi[tower_idx]/math.pi,\
                                     0.0,
                                     0.0]
                if not use_jetPt:
                    tower_feature = [tower_feature[0]] + tower_feature[2:]
                    len_tower_attr -= 1
                if not with_node_type:
                    tower_feature = tower_feature[1:]
                    len_tower_attr -= 1
                
                tower_nodes.append(tower_feature)

            tower_idx += 1
        
        tower_nodes.sort(reverse=True)
        if tower_lim != None:
            tower_nodes = tower_nodes[0:min(len(tower_nodes),tower_lim)]
            pad_array(tower_nodes, tower_lim, len_tower_attr)
            

        split_point = len(tower_nodes)
        
        len_track_attr = 7
        for itrack in range(chain.JetGhostTrackN[ijet]):
            ghost_track_idx = chain.JetGhostTrackIdx[track_idx]
            
            if(not cutoff or chain.TrackPt[ghost_track_idx] >= 1.0):
                deta = chain.JetEta[ijet]-chain.TrackEta[ghost_track_idx]
                dphi = calc_dphi(chain.JetPhi[ijet],chain.TrackPhi[ghost_track_idx])
                theta = 2*math.atan(-math.exp(chain.TrackEta[ghost_track_idx]))
                z0 = math.log10(10e-3+math.fabs(chain.TrackZ0[ghost_track_idx]*math.sin(theta)))
                d0 = math.log10(10e-3+math.fabs(chain.TrackD0[ghost_track_idx]))
                
                if use_delta_angles:
                    track_feature = [1.0, math.log10(chain.JetPt[ijet]),
                                     math.log10(chain.TrackPt[ghost_track_idx]),
                                     math.fabs(deta),
                                     math.fabs(dphi),
                                     z0,
                                     d0]
                else:
                    track_feature = [1.0, math.log10(chain.JetPt[ijet]),\
                                     math.log10(chain.TrackPt[ghost_track_idx]),\
                                     chain.TrackEta[ghost_track_idx]/3,\
                                     chain.TrackPhi[ghost_track_idx]/math.pi,\
                                     z0,\
                                     d0]
                if not use_jetPt:
                    track_feature = [track_feature[0]] + track_feature[2:]
                    len_track_attr -= 1
                if not with_node_type:
                    track_feature = track_feature[1:]
                    len_track_attr -= 1
                track_nodes.append(track_feature)
            track_idx+=1

        track_nodes.sort(reverse=True)
        if track_lim != None:
            track_nodes = track_nodes[0:min(len(track_nodes),track_lim)]
            pad_array(track_nodes,track_lim,len_track_attr)
            

        nodes = np.array(tower_nodes + track_nodes,dtype=np.float32)
        node_type = np.array([0] * len(tower_nodes) + [1] * len(track_nodes), dtype=np.int8)

        n_nodes = len(nodes)
        
        # Filtering out jets
        if n_nodes <= 1:
            continue
        if n_nodes == 2 and split_point < 1:
            continue
        if chain.JetPt[ijet] < 30 or abs(chain.JetEta[ijet]) > 3:
            continue
        if signal != 0 and isTau == 0:
            continue
        elif signal == 0 and isTau:
            continue
        if signal != None and signal != 0 and signal != 10 and signal != isTau:
            continue
        

        if debug:
            #print(nodes.shape)
            #print(n_nodes)
            #print(nodes)
            pass
     
        ### edges ###
        if connectivity == 'disconnected':
            all_edges = list(itertools.combinations(range(split_point), 2)) + list(itertools.combinations(range(split_point, n_nodes), 2))
        elif connectivity == 'KNN':
            nbrs = NearestNeighbors(n_neighbors=3).fit(nodes)
            distances, indices = nbrs.kneighbors(nodes)
            all_edges = indices
        elif connectivity == 'sequential':
            all_edges = [(i, i+1) for i in range(split_point-1)] + [(i, i+1) for i in range(split_point, n_nodes-1)]
        else:
            all_edges = list(itertools.combinations(range(n_nodes), 2))
        senders = np.array([x[0] for x in all_edges])
        receivers = np.array([x[1] for x in all_edges])
        n_edges = len(all_edges)
        
            
        edges = []        
        Delta = lambda ya, yb, phiA, phiB: np.sqrt((ya-yb)**2+(phiA-phiB)**2)
        kT = lambda ptA, ptB, delta: min(ptA, ptB) * delta
        Z = lambda ptA, ptB: min(ptA, ptB) / (ptA + ptB)
        M2 = lambda Pt1, Pt2, eta1, eta2, phi1, phi2: 2*Pt1*Pt2*(np.cosh(eta1-eta2)-np.cos(calc_dphi(phi1, phi2)))
        
        
        edge_starting = 2 if with_node_type else 1
        for e in all_edges:
            v1, v2 = nodes[e[0]], nodes[e[1]]
            edge_feature = []
            if with_edge_features:
                
                pt1, eta1, phi1 = v1[edge_starting:edge_starting+3]
                pt2, eta2, phi2 = v2[edge_starting:edge_starting+3]
                delta = Delta(eta1, eta2, phi1, phi2)
                kt = kT(10**pt1, 10**pt2, delta)
                z = Z(10**pt1, 10**pt2)
                m2 = M2(10**pt1, 10**pt2, eta1, eta2, phi1, phi2)
                if m2 <= 0:
                    m2 = 1e-10
                edge_feature = [np.log(delta), np.log(kt), np.log(z), np.log(m2)]
            # Determine Edge Types
            if node_type[e[0]] > 0.5:
                if node_type[e[1]] > 0.5: # track-track
                    edge_type = [0.0, 0.0, 1.0]
                else: #track-cluster should not be possible
                    raise("Error: Tracks and clusters out of order")
            elif node_type[e[1]] > 0.5: # cluster-track
                edge_type = [0.0, 1.0, 0.0]
            else: # cluster-cluster
                edge_type = [1.0, 0.0, 0.0]
            
            edges.append(edge_type + edge_feature)
            
            
        ### Globals ###
        if use_jetVar:
            global_features = [chain.JetPt[ijet], chain.JetEta[ijet], chain.JetPhi[ijet]]
        else:
            global_features = [n_nodes]
        if with_hlv_features:
            global_features += [chain.JetLeadingTrackFracP[ijet],
                                chain.JetTrackR[ijet],
                                chain.JetNumISOTracks[ijet],
                                chain.JetMaxDRInCore[ijet],
                                chain.JetTrackMass[ijet]]
            
        
        nodes = np.array(nodes,dtype=np.float32)
        edges = np.array(edges, dtype=np.float32)
        globals = np.array(global_features, dtype=np.float32)  
        if n_edges < 1:
            edges = np.expand_dims(np.array([0.0]*n_edges, dtype=np.float32), axis=1)
        

        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": globals
        }
        target_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([1. if isTau else 0.],dtype=np.float32)
        }
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        graph_list.append((input_graph, target_graph))
    #print(target_graph)
    #exit()
    if len(graph_list) == 0:
        return [(None, None)]
    
    return graph_list


def read(filename, start_entry=0, nentries=float('inf')):
    import ROOT
    chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
    chain.Add(filename)
    tot_entries = chain.GetEntries()
    nentries = nentries if (start_entry + nentries) <= tot_entries\
        else tot_entries - start_entry
    #print("Total {:,} Events".format(nentries))
    for ientry in range(nentries):
        chain.GetEntry(ientry + start_entry)
        yield chain

class TauIdentificationDataset(DataSet):
    """
    Tau Identification dataset with heterogeneous nodes. The graph can be set to be fully-connected, fully-connected only among the same type of nodes (denoted as 'disconnected'), or KNN based. 
    Each node in the graph is a vector of length 7, with components:
        * 1 if track, 0 if tower
        * Pt of the jet
        * Et of tower/Pt of track
        * Eta of tower/track
        * Phi of tower/track
        * d0 of track, 0 if tower
        * z0 of track, 0 if tower
    Each edge is a vector of length 7, with components:
        * one-hot encoding for the type of edges
        * 4 edge features, delta, kT, z, mass square
    """
    def __init__(self):
        super().__init__()
        self.read = read
        self.make_graph = make_graph

    def set_config_file(self,config):
        config = load_yaml(config)
        self.tower_lim = config.get('tower_limit',None)
        self.track_lim = config.get('track_limit',None)
        self.use_cutoff = config.get('use_cutoff',False)

    def _num_evts(self, filename):
        import ROOT
        chain = ROOT.TChain(tree_name, tree_name) # pylint: disable=maybe-no-member
        chain.Add(filename)
        n_entries = chain.GetEntries()
        return n_entries

    
    def subprocess(self, ijob, n_evts_per_record, filename, outname, overwrite, debug, **kwargs):
       
        outname = "{}_{}.tfrec".format(outname, ijob)
        if os.path.exists(outname) and not overwrite:
            print(outname,"is there. skip...")
            return 0, n_evts_per_record

        ifailed = 0
        all_graphs = []
        start_entry = ijob * n_evts_per_record
        
        if debug:
            print(">>> Debug 0", ijob)
        
        t0 = time.time()
        jevt = 0
        kgraphs = 0
        for event in self.read(filename, start_entry, n_evts_per_record):
            gen_graphs = self.make_graph(event, debug, **kwargs)
            
            if debug:
                print(">>> Debug 1", ijob, jevt, kgraphs)
            
            if len(gen_graphs)==0 or gen_graphs[0][0] == None:
                ifailed += 1
                continue

            all_graphs += gen_graphs
            kgraphs += len(gen_graphs)
            jevt += 1
            
        if debug:
            print(">>> Debug 2", ijob)
            
        isaved = len(all_graphs)
        if isaved > 0:
            ex_input, ex_target = all_graphs[0]
            input_dtype, input_shape = graph.dtype_shape_from_graphs_tuple(
                ex_input, with_padding=self.with_padding)
            target_dtype, target_shape = graph.dtype_shape_from_graphs_tuple(
                ex_target, with_padding=self.with_padding)
            def generator():
                for G in all_graphs:
                    yield (G[0], G[1])
            

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=(input_dtype, target_dtype),
                output_shapes=(input_shape, target_shape),
                args=None)
            if debug:
                print(">>> Debug 3", ijob)
            writer = tf.io.TFRecordWriter(outname)
            for data in dataset:
                example = graph.serialize_graph(*data)
                writer.write(example)
            if debug:
                print(">>> Debug 4", ijob)
            writer.close()
            t1 = time.time()
            all_graphs = []
            print(f">>> Job {ijob} Finished in {abs(t1-t0)/60:.2f} min")
        else:
            print(ijob, "all failed")
        return ifailed, isaved

    def process(self, filename, outname, n_evts_per_record,
        debug, max_evts, num_workers=1, overwrite=False, **kwargs):
        
        rng = np.random.default_rng(12345)

        now = time.time()

        all_evts = self._num_evts(filename)
        all_evts = max_evts if max_evts > 0 and all_evts > max_evts else all_evts

        n_files = all_evts // n_evts_per_record
        if all_evts%n_evts_per_record > 0:
            n_files += 1

        print("Total {:,} events are requested to be written to {:,} files with {:,} workers".format(all_evts, n_files, num_workers))
        out_dir = os.path.abspath(os.path.dirname(outname))
        os.makedirs(out_dir, exist_ok=True)
        
        if num_workers < 2:
            ifailed, isaved=0, 0
            for ijob in range(n_files):
                n_failed, n_saved = self.subprocess(
                    ijob, n_evts_per_record, filename, outname, 
                    overwrite, debug, rand=rng, **kwargs)
                ifailed += n_failed
                isaved += n_saved
        else:
            with Pool(num_workers) as p:
                process_fnc = partial(self.subprocess,
                        n_evts_per_record=n_evts_per_record,
                        filename=filename,
                        outname=outname,
                        overwrite=overwrite,
                        debug=debug,
                        rand=rng,
                        **kwargs)
                res = p.map(process_fnc, list(range(n_files)))

            ifailed = sum([x[0] for x in res])
            isaved = sum([x[1] for x in res])
            
        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            isaved, read_time/60.))
        print("{:,} events failed in being converted to graph".format(ifailed))
        