import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Tensorflow/Keras implementation of "Consistent Rank Logits" for deep ordinal regression, or CORAL, including
# output layer and loss function. Original paper: Cao et al. (2019) https://arxiv.org/abs/1901.07884

# Coral output layer: a Dense layer outputting to one node, then independent biases are added to that node
# for the output layer. Provides logits for the loss function and probits (sigmoid) otherwise.
class CoralOutput(tf.keras.layers.Layer):
  def __init__(self, output_len):
    super(CoralOutput, self).__init__()
    self.output_len = output_len

  def build(self, input_shape):
    self.kernel = self.add_weight(
        'kernel',
        shape=[int(input_shape[-1]), 1],
        initializer='glorot_uniform',
        dtype='float32',
        trainable=True
    )
    self.biases = self.add_weight(
        'biases',
        shape=[self.output_len,],
        initializer='zeros',
        dtype='float32',
        trainable=True
    )

  def call(self, input):
    fc = tf.matmul(input, self.kernel)
    fc = tf.tile(fc, [1, self.output_len])
    logits = tf.nn.bias_add(fc, self.biases, name='logits')
    probits = tf.math.sigmoid(logits, name='probits')
    return logits, probits


# Implementation of https://github.com/Raschka-research-group/coral-cnn/blob/master/model-code/resnet34/cacd-coral.py#L326
# The authors describe this as "weighted cross-entropy" of the K-1 binary classifiers. Note their code implentation
# differs from the form in the paper for numerical stability.
# More info in https://github.com/Raschka-research-group/coral-cnn/issues/9
def coral_loss(all_y=None):
    importance = 1 if all_y is None else _task_importance_weighting(all_y)
    def loss_logits(y_true, y_pred):
        unweighted = (tf.math.log_sigmoid(y_pred) * y_true) + (tf.math.log_sigmoid(y_pred) - y_pred) * (1 - y_true)
        return tf.reduce_mean(-1 * tf.reduce_sum(importance * unweighted, axis=1))
    return loss_logits


# Optionally weight the loss per-class based on the authors' "task importance weighting" method.
# Lesser-balanced classes are weighted greater loss.
def _task_importance_weighting(y):
    n_ranks = y.shape[-1]
    ranks = 1 + np.sum(y, axis=-1)
    n_examples = len(y)
    m = np.zeros(n_ranks)
    for k in range(n_ranks):
        s_k = np.sum(ranks > (k + 1))
        m[k] = np.sqrt(max(s_k, n_examples - s_k))
    return (m / np.max(m))