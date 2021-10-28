import numpy as np
import tensorflow as tf


from root_gnn.trainer import Trainer

def augument(jet_graph):
    # the contintuents of jet should be transformed w.r.t the jet axis (eta, phi)
    # jet graph must have global attributes of [pt, eta, phi]

    batch_size = jet_graph.globals.shape[0]

    #<TODO> https://numpy.org/doc/stable/reference/random/generator.html
    # 1) Use a seed in the random rotation so one can reproduce the results.
    # 2) sample <batch_size> size..
    theta = np.random.uniform(0,2*np.pi)
    shift_eta = np.random.uniform(-1,1)
    shift_phi = np.random.uniform(-1,1)

    new_eta = jet_graph.globals[0][1] + shift_eta
    new_phi = jet_graph.globals[0][2] + shift_phi

    new_globals = [jet_graph.globals[0][0], new_eta, new_phi]
    nodes = jet_graph.nodes
    new_nodes = []
    for node_idx in range(jet_graph.n_node):
        new_node_eta = nodes[node_idx][1]*np.cos(theta) - nodes[node_idx][2]*np.sin(theta) +shift_eta
        new_node_phi = nodes[node_idx][1]*np.sin(theta) + nodes[node_idx][2]*np.cos(theta)+shift_phi
        new_nodes.append([jet_graph.nodes[node_idx][0], new_node_eta, new_node_phi])
    new_graph = jet_graph.replace(nodes=new_nodes, globals=new_globals)
    return new_graph 


class EmbeddingTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def _setup_training_loop(self):
        if self.training_step:
            return 
        
        def update_step(self, inputs):
            print("Tracing update_step")
            aug_inputs = augument(inputs)

            with tf.GradientTape() as tape:
                
                output_ops = self.model(inputs, self.num_iters)
                aug_output_ops = self.model(aug_inputs, self.num_iters)
                loss_ops_tr = self.loss_fcn(output_ops, aug_output_ops)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return loss_op_tr

        self.training_step = tf.function(update_step)