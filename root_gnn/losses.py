import tensorflow as tf

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


if __name__ == "__main__":
    node_edge_loss = NodeEdgeLoss(2, 1)
    node_edge_loss(1, 1)