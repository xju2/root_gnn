import tensorflow as tf
import sonnet as snt
#import tensorflow_addons as tfa
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_nets import utils_tf
from tensorflow.python.ops import math_ops


class NodeEdgeLoss:
    def __init__(self, real_edge_weight, fake_edge_weight,
                real_node_weight, fake_node_weight):
        self.w_node_real = real_node_weight
        self.w_node_fake = fake_node_weight
        self.w_edge_real = real_edge_weight
        self.w_edge_fake = fake_edge_weight

    def __call__(self, target_op, output_ops):
        node_weights = target_op.nodes * self.w_node_real + (1 - target_op.nodes) * self.w_node_fake
        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.nodes, output_op.nodes, weights=node_weights)
                for output_op in output_ops
        ]
        edge_weights = target_op.edges * self.w_edge_real + (1 - target_op.edges) * self.w_edge_fake
        loss_ops += [
            tf.compat.v1.losses.log_loss(target_op.edges, output_op.edges, weights=edge_weights)
                for output_op in output_ops 
        ]
        return tf.stack(loss_ops)


class GlobalLoss:
    def __init__(self, real_global_weight, fake_global_weight):
        self.w_global_real = real_global_weight
        self.w_global_fake = fake_global_weight

    def __call__(self, target_op, output_ops):
        global_weights = target_op.globals * self.w_global_real \
            + (1 - target_op.globals) * self.w_global_fake
        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.globals, output_op.globals, weights=global_weights)
            for output_op in output_ops
        ]
        return tf.stack(loss_ops)

class RegressionLoss(snt.Module):
    """
    Loss functions for regression.
    Supported loss function name: 
    * AME, absolute mean error
    * MSE, mean squared error
    """
    def __init__(self, loss_name: str=None, name='RegressionLoss') -> None:
        super().__init__(name=name)
        name = name.lower()
        
        self.fnc = tf.compat.v1.losses.absolute_difference
        if loss_name and loss_name == 'mse':
            self.fnc = tf.compat.v1.losses.mean_squared_error

class GlobalRegressionLoss(RegressionLoss):
    def __init__(self, loss_name: str = None, name: str = "GlobalRegressionLoss") -> None:
        super().__init__(loss_name=loss_name, name="GlobalRegressionLoss")

    def __call__(self, target_op, output_ops):
        loss_ops = [
            self.fnc(target_op.globals, output_op.globals) for output_op in output_ops
        ]
        return tf.stack(loss_ops)
    
    
class RegressionRepLoss(GlobalRegressionLoss):
    def __init__(self, loss_name: str = None, name: str = "GlobalRepresentationLoss") -> None:
        super().__init__(loss_name='mse', name="GlobalRepresentationLoss")
    
    def __call__(self, output_ops, aug_output_ops, _):
        loss_ops = [
            self.fnc(op[0].globals, op[1].globals) for op in zip(output_ops, aug_output_ops)
        ]
        return tf.stack(loss_ops)
    

class ClassificationRepLoss(GlobalLoss):
    def __init__(self, real_global_weight, fake_global_weight):
        super().__init__(real_global_weight, fake_global_weight)
        self.log = tf.compat.v1.losses.log_loss
        
    
    def __call__(self, output_ops, aug_output_ops, target_op):
        op1 = [self.log(target_op.globals, output_op.globals) for output_op in output_ops]
        temperature = 0.1
        op2 = []
        
        def cos_sim(y_true, y_pred, axis=-1):
            y_true = tf.nn.l2_normalize(y_true, axis=axis)
            y_pred = tf.nn.l2_normalize(y_pred, axis=axis)
            return -math_ops.reduce_sum(y_true * y_pred, axis=axis)
        
        for op in zip(output_ops, aug_output_ops):
            x_i, x_j = op[0].globals, op[1].globals
            batch_size = x_i.shape[0]
            z_i = tf.linalg.normalize(x_i, axis=1)[0]
            z_j = tf.linalg.normalize(x_j, axis=1)[0]
            z = tf.concat([z_i, z_j], axis=0)

            similarity_matrix = cos_sim(tf.expand_dims(z, 1), tf.expand_dims(z, 0), axis=2)
            sim_ij = tf.linalg.diag_part(similarity_matrix, k=batch_size)
            sim_ji = tf.linalg.diag_part(similarity_matrix, k=-batch_size)
            positives = tf.concat( [sim_ij, sim_ji], axis=0 )
            nominator = tf.math.exp( positives / temperature )
            
            I = tf.eye(2*batch_size, 2*batch_size, dtype=tf.dtypes.float32)
            negatives_mask = tf.ones([2*batch_size, 2*batch_size], dtype=tf.dtypes.float32) - I
            denominator = tf.math.multiply(negatives_mask, tf.math.exp(similarity_matrix / temperature))

            loss_partial = -tf.math.log( nominator / tf.math.reduce_sum( denominator, axis=1 ) )
            loss = tf.math.reduce_sum( loss_partial )/( 2*batch_size )
            op2.append(loss)
        
        loss_ops = op1 + op2
        return tf.stack(loss_ops)
    
    
class ContrastiveLoss(GlobalRegressionLoss):
    def __init__(self, loss_name: str = None, name: str = "ContrastiveLoss") -> None:
        super().__init__(loss_name=loss_name, name="ContrastiveLoss")
        self.fnc = tfa.losses.contrastive_loss
        
    def __call__(self, output_ops, aug_output_ops, _):
        loss_ops = [
            self.fnc(op[0].globals, op[1].globals) for op in zip(output_ops, aug_output_ops)
        ]
        return tf.stack(loss_ops)
    
    
class RepContrastiveLoss(snt.Module):
    def __init__(self, a, b, name: str = "RepContrastiveLoss") -> None:
        super().__init__(name=name)
        
    def __call__(self, output_ops, aug_output_ops, _):
        temperature = 0.1
        loss_ops = []
        
        def cos_sim(y_true, y_pred, axis=-1):
            y_true = tf.nn.l2_normalize(y_true, axis=axis)
            y_pred = tf.nn.l2_normalize(y_pred, axis=axis)
            return -math_ops.reduce_sum(y_true * y_pred, axis=axis)
        
        for op in zip(output_ops, aug_output_ops):
            x_i, x_j = op[0].globals, op[1].globals
            batch_size = x_i.shape[0]
            z_i = tf.linalg.normalize(x_i, axis=1)[0]
            z_j = tf.linalg.normalize(x_j, axis=1)[0]
            z = tf.concat([z_i, z_j], axis=0)

            similarity_matrix = cos_sim(tf.expand_dims(z, 1), tf.expand_dims(z, 0), axis=2)
            sim_ij = tf.linalg.diag_part(similarity_matrix, k=batch_size)
            sim_ji = tf.linalg.diag_part(similarity_matrix, k=-batch_size)
            positives = tf.concat( [sim_ij, sim_ji], axis=0 )
            nominator = tf.math.exp( positives / temperature )
            
            I = tf.eye(2*batch_size, 2*batch_size, dtype=tf.dtypes.float32)
            negatives_mask = tf.ones([2*batch_size, 2*batch_size], dtype=tf.dtypes.float32) - I
            denominator = tf.math.multiply(negatives_mask, tf.math.exp(similarity_matrix / temperature))

            loss_partial = -tf.math.log( nominator / tf.math.reduce_sum( denominator, axis=1 ) )
            loss = tf.math.reduce_sum( loss_partial )/( 2*batch_size )
            loss_ops.append(loss)
            
        return tf.stack(loss_ops)
        
    


class EdgeRegressionLoss(RegressionLoss):
    def __init__(self, loss_name: str = None, name: str = "EdgeRegressionLoss") -> None:
        super().__init__(loss_name=loss_name, name=name)
    
    def __call__(self, target_op, output_ops):
        loss_ops = [
            self.fnc(target_op.edges, output_op.edges) for output_op in output_ops
        ]
        return tf.stack(loss_ops)        


class EdgeGlobalLoss:
    def __init__(self, real_edge_weight, fake_edge_weight,
                real_global_weight, fake_global_weight):
        self.w_edge_real = real_edge_weight
        self.w_edge_fake = fake_edge_weight
        self.w_global_real = real_global_weight
        self.w_global_fake = fake_global_weight

    def __call__(self, target_op, output_ops):
        global_weights = target_op.globals * self.w_global_real \
            + (1 - target_op.globals) * self.w_global_fake
        edge_weights = target_op.edges * self.w_edge_real \
            + (1 - target_op.edges) * self.w_edge_fake

        loss_ops = [
            tf.compat.v1.losses.log_loss(target_op.globals, output_op.globals, weights=global_weights)
            for output_op in output_ops
        ]
        loss_ops += [
            tf.compat.v1.losses.log_loss(target_op.edges, output_op.edges, weights=edge_weights)
            for output_op in output_ops
        ]
        return tf.stack(loss_ops)

class EdgeLoss:
    def __init__(self, real_edge_weight, fake_edge_weight):
        self.w_edge_real = real_edge_weight
        self.w_edge_fake = fake_edge_weight

    def __call__(self, target_op, output_ops):
        t_edges = tf.squeeze(target_op.edges)
        edge_weights = t_edges * self.w_edge_real + (1 - t_edges) * self.w_edge_fake
        loss_ops = [
            tf.compat.v1.losses.log_loss(t_edges, tf.squeeze(output_op.edges), weights=edge_weights) 
                for output_op in output_ops 
        ]
        return tf.stack(loss_ops)
    



__all__ = (
    "NodeEdgeLoss",
    "GlobalLoss",
    "EdgeGlobalLoss",
    "EdgeLoss",
    "GlobalRegressionLoss",
    "GlobalRepresentationLoss",
    "EdgeRegressionLoss",
    "ContrastiveLoss",
    "ClassificationRepLoss",
    "RepContrastiveLoss"
)

if __name__ == "__main__":
    node_edge_loss = NodeEdgeLoss(2, 1, 2, 1)
    node_edge_loss(1, 1)
