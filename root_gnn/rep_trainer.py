import numpy as np
import tensorflow as tf

from root_gnn.homo_trainer import *



def rotation(jet_graph, theta, eta, phi, debug=False):
    """
    Do rotation on each jet graph's nodes using the theta, eta, phi given.
    """
    nodes = jet_graph.nodes
    if debug:
        print("**********", jet_graph, "\n", int(jet_graph.n_node), "**********")

    rot = tf.constant([[1, 0, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0, 0],
                       [0, 0, np.cos(theta), np.sin(theta), 0, 0, 0],
                       [0, 0, -np.sin(theta),  np.cos(theta), 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 1]])
    shift = tf.constant([0, 0, eta, phi, 0, 0, 0])


    new_nodes = tf.linalg.matmul(nodes, rot)
    new_nodes = tf.math.add(new_nodes, shift)
    new_graph = jet_graph.replace(nodes=new_nodes)
    return new_graph


def ghostTracks(jet_graph, pt, eta, phi, debug=False):
    """
    Must have global information with (eta, phi) of the jet.
    NOTE: THIS FUNCTION ACTUALLY HAS ISSUES DURING TRAINING SINCE IT ONLY ADDS NODES AND CANNOT GIVE ANY CONNECTIVITY INFO. CONSIDER CHANGING.
    """
    if debug:
        print("***************** DEBUG MODE *********************")
        print(f">>> Globals: {jet_graph.globals}")
        print(f">>> Globals components: {jet_graph.globals[0]}")
        #exit()
        
    jet_eta, jet_phi = jet_graph.globals[0][0]/3., jet_graph.globals[0][1]/np.pi
    factor = np.sqrt(1/2)*0.4 
    
    if debug:
        print(jet_eta)
    
    allEta = (factor * eta ) / 3. + jet_eta
    allPhi = (factor * phi ) / np.pi + jet_phi
    pt = pt/1e3
    nodes = jet_graph.nodes
    if debug:
        print(f">>> Node shape: {utils_tf._get_shape(nodes)}")
        print(f">>> Nodes[0]: {nodes[0]}")
        print(f">>> Nodes[9]: {nodes[9]}")
        print(f">>> Nodes: {nodes[0:5]}")
        #exit()
        
    new_nodes = tf.stack([pt, allEta, allPhi], axis=1)
    new_nodes = tf.concat([new_nodes, nodes], axis=0)
    
    if debug:
        print(f">>> New nodes {new_nodes}")
        #exit()
        
    new_graph = jet_graph.replace(nodes=new_nodes)
    return new_graph
    
    
def augument(jet_graphs, batch_size, augment_type="rotation", seed=12345, debug=False):
    """
    the contintuents of jet should be transformed w.r.t the jet axis (eta, phi)
    jet graph must have global attributes of [pt, eta, phi]
    """
    if debug:
        batch_size = 1
        
    rng = np.random.default_rng(seed)
    if augment_type == "rotation":
        theta = rng.random(batch_size)*2*np.pi
        shift_eta = rng.random(batch_size)*2-1
        shift_phi = rng.random(batch_size)*2-1
    elif augment_type == "ghost_tracks":
        num_ghost_tracks = rng.integers(low=1, high=12, size=batch_size)
        pt = rng.random(batch_size*12) * 3
        eta = rng.random(batch_size*12) * 2 - 1
        phi = rng.random(batch_size*12) * 2 - 1
    
    if debug:
        print(jet_graphs)
    
    new_graphs = []
    
    for graph_id in tqdm.trange(batch_size, desc="Augmenting Graphs", disable=True):
        jet_graph = utils_tf.get_graph(jet_graphs, graph_id)
        if augment_type == "rotation":
            new_graph = rotation(jet_graph, theta[graph_id], shift_eta[graph_id], shift_phi[graph_id])
        elif augment_type == "ghost_tracks": 
            n = num_ghost_tracks[graph_id]
            i = graph_id * 12
            new_graph = ghostTracks(jet_graph, pt[i:i+n]/n, eta[i:i+n], phi[i:i+n])
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
            
            aug_input_list = []
            for _ in range(self.num_transformation):
                if self.augment_type == "ghost_tracks":
                    aug_input = targets
                else:
                    aug_input = augument(inputs, self.batch_size, augment_type=self.augment_type)
                aug_input_list.append(aug_input)
                
            print(f">>Augumented {self.num_transformation} Time(s) Successfully Using {self.augment_type}>>")
            
            with tf.GradientTape() as tape:
                output_ops = self.model(inputs, self.num_iters)
                aug_output_ops_list = []
                for aug_inputs in aug_input_list:
                    aug_output_ops = self.model(aug_inputs, self.num_iters)
                    aug_output_ops_list.append(aug_output_ops)
                loss_ops_tr = self.loss_fcn(output_ops, aug_output_ops_list, targets)
                loss_op_tr = tf.math.reduce_sum(loss_ops_tr) / tf.constant(self.num_iters, dtype=tf.float32)

            gradients = tape.gradient(loss_op_tr, self.model.trainable_variables)
            
            #receive learning rate from schedule
            if self.cosine_decay:
                lr_mult = self.get_lr(step)
                self.lr.assign(self.lr_base * lr_mult)
                if step % self.decay_steps == 0 and step / self.decay_steps > 0:
                    ckpt_n = step / self.decay_steps
                    ckpt_dir = os.path.join(output_dir, "one-shot-checkpoints/{ckpt_n}")
                    self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model)
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
        predictions, truth_info, augment_outs = [], [], []
        for _ in range(self.val_batches):
            inputs, targets = next(val_data)
            if self.augment_type == "ghost_tracks":
                aug_inputs = targets
            else:
                aug_inputs = augument(inputs, self.batch_size, augment_type=self.augment_type)
            
            outputs = self.model(inputs, self.num_iters, is_training=False)
            aug_output_ops = self.model(aug_inputs, self.num_iters, is_training=False)
            
            total_loss += (tf.math.reduce_sum(
                self.loss_fcn(outputs, [aug_output_ops], targets))/tf.constant(
                    self.num_iters, dtype=tf.float32)).numpy()
            if len(outputs) > 1:
                outputs = outputs[-1]
                aug_output_ops = aug_output_ops[-1]
            if type(outputs) == list:
                outputs = outputs[-1]
                aug_output_ops = aug_output_ops[-1]

            if "globals" in self.mode:
                predictions.append(outputs.globals)
                truth_info.append(targets.globals)
                augment_outs.append(aug_output_ops.globals)
                
            elif 'edges' in self.mode:
                predictions.append(outputs.edges)
                truth_info.append(targets.edges)
            else:
                raise ValueError("currently " + self.mode + " is not supported")

        predictions = np.concatenate(predictions, axis=0)
        
        if 'rgr' in self.mode:
            truth_info = np.concatenate(augment_outs, axis=0)
        else:
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
    
    