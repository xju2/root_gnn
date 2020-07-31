import tensorflow as tf
import sonnet as snt

from graph_nets import utils_tf
from graph_nets import modules

from root_gnn.src.models.base import MLPGraphIndependent
from root_gnn.src.models.base import MLPGraphNetwork

LATENT_SIZE = 128

class GeneralClassifier(snt.Module):

    def __init__(self, name="GeneralClassifier"):
        super(GeneralClassifier, self).__init__(name=name)

        self._encoder = MLPGraphIndependent()
        self._core    = MLPGraphNetwork()
        self._decoder = MLPGraphIndependent()

        # Transforms the outputs into appropriate shapes.
        global_output_size = 1
        global_fn =lambda: snt.Sequential([
            snt.nets.MLP([LATENT_SIZE, global_output_size],
                         name='global_output'), tf.sigmoid])

        self._output_transform = modules.GraphIndependent(None, None, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent

        output_ops = []
        for _ in range(num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)

        decoded_op = self._decoder(latent)
        output_ops.append(self._output_transform(decoded_op))
        return output_ops