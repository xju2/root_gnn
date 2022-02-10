import numpy as np
import tensorflow as tf

from root_gnn.trainer import *



def rotation(jet_graph, theta, eta, phi, debug=False):
        """
        Do rotation on each jet graph's nodes using the theta, eta, phi given.
        """
        nodes = jet_graph.nodes
        if debug:
            print("**********", jet_graph, "\n", int(jet_graph.n_node), "**********")

        rot = tf.constant([[1, 0, 0],
                           [0, np.cos(theta), np.sin(theta)],
                           [0, -np.sin(theta),  np.cos(theta)]])
        shift = tf.constant([0, eta, phi])
        
        
        new_nodes = tf.linalg.matmul(nodes, rot)
        new_nodes = tf.math.add(new_nodes, shift)
        new_graph = jet_graph.replace(nodes=new_nodes)
        return new_graph
    
    
def augument(jet_graphs, batch_size, seed=12345, debug=False):
    """
    the contintuents of jet should be transformed w.r.t the jet axis (eta, phi)
    jet graph must have global attributes of [pt, eta, phi]
    """
    if debug:
        batch_size = 1
        
    rng = np.random.default_rng(seed)
    theta = rng.random(batch_size)*2*np.pi
    shift_eta = rng.random(batch_size)*2-1
    shift_phi = rng.random(batch_size)*2-1
    
    if debug:
        print(jet_graphs)
    
    new_graphs = []
    
    for graph_id in range(batch_size):
        jet_graph = utils_tf.get_graph(jet_graphs, graph_id)
        nodes = jet_graph.nodes
        new_graph = rotation(jet_graph, theta[graph_id], shift_eta[graph_id], shift_phi[graph_id])
        new_graphs.append(new_graph)
        
    if debug:  
        print(new_graphs[0])

    new_graphs_tr = utils_tf.concat(new_graphs, axis=0)
    return new_graphs_tr
    



class EmbeddingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
    def _setup_training_loop(self):
        
        
        if self.training_step:
            return 
        
        
        input_signature = get_signature(self.data_train)
        
        def update_step(inputs, targets):
            print(">>Tracing update_step>>")
            
            aug_inputs = augument(inputs, self.batch_size)
            print(">>Augumented Successfully>>")
            
            with tf.GradientTape() as tape:
                output_ops = self.model(inputs, self.num_iters)
                aug_output_ops = self.model(aug_inputs, self.num_iters)
                loss_ops_tr = self.loss_fcn(output_ops, aug_output_ops, targets)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, self.model.trainable_variables)
            self.optimizer.apply(gradients, self.model.trainable_variables)
            return loss_op_tr

        self.training_step = tf.function(update_step, input_signature=input_signature)
        
    def validation(self):
        """
        Performs validation steps, record performance metrics.
        All is based on `mode`. 
        """
        val_data = self.data_val

        total_loss = 0.
        predictions, truth_info = [], []
        for _ in range(self.val_batches):
            inputs, targets = next(val_data)
            aug_inputs = augument(inputs, self.batch_size)
            
            outputs = self.model(inputs, self.num_iters, is_training=False)
            aug_output_ops = self.model(aug_inputs, self.num_iters, is_training=False)
            
            total_loss += (tf.math.reduce_sum(
                self.loss_fcn(outputs, aug_output_ops, targets))/tf.constant(
                    self.num_iters, dtype=tf.float32)).numpy()
            if len(outputs) > 1:
                outputs = outputs[-1]
            if type(outputs) == list:
                outputs = outputs[-1]

            if "globals" in self.mode:
                predictions.append(outputs.globals)
                truth_info.append(targets.globals)
            elif 'edges' in self.mode:
                predictions.append(outputs.edges)
                truth_info.append(targets.edges)
            else:
                raise ValueError("currently " + self.mode + " is not supported")

        predictions = np.concatenate(predictions, axis=0)
        truth_info = np.concatenate(truth_info, axis=0)
        
        if 'clf' in self.mode:
            threshold = 0.5
            y_true, y_pred = (truth_info > threshold), (predictions > threshold)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, predictions)
            self.metric_dict['auc'] = sklearn.metrics.auc(fpr, tpr)
            self.metric_dict['acc'] = sklearn.metrics.accuracy_score(y_true, y_pred)
            self.metric_dict['pre'] = sklearn.metrics.precision_score(y_true, y_pred)
            self.metric_dict['rec'] = sklearn.metrics.recall_score(y_true, y_pred)
            value = self.metric_dict['acc']
            
        elif 'rgr' in self.mode:
            self.metric_dict['pull'] = np.mean((predictions - truth_info) / truth_info)
            value = self.metric_dict['pull']
        else:
            raise ValueError("currently " + self.mode + " is not supported")

        self.metric_dict['val_loss'] = total_loss / self.val_batches

        with self.metric_writer.as_default():
            for key,val in self.metric_dict.items():
                tf.summary.scalar(key, val, step=self.step_count)


        current_metric = self.metric_dict[self.stop_on]
        is_better = (self.should_max_metric and current_metric > self.best_metric) \
            or (not self.should_max_metric and current_metric < self.best_metric)
        if is_better:
            self.best_metric = current_metric
        return value, total_loss
    
    
