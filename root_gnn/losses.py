import tensorflow as tf
import sonnet as snt

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
        super().__init__(loss_name=loss_name, name=name)

    def __call__(self, target_op, output_ops):
        loss_ops = [
            self.fnc(target_op.globals, output_op.globals) for output_op in output_ops
        ]
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
    "EdgeRegressionLoss"
)

if __name__ == "__main__":
    node_edge_loss = NodeEdgeLoss(2, 1, 2, 1)
    node_edge_loss(1, 1)