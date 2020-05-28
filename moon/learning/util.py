import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

'''
Metrics and other utilities for ordinal regression and classification. For OrdReg, a primary
assumption is that rank1 is represented by [0, 0, ...], rank2 by [1, 0, ...], and so on. For
classification, a primary assumption is each rank is represented with a one-hot.
'''

# For some background on "deep ordinal regression/classification"...
# Cheng (2007) proposes independent sigmoids, with MSE loss (on the vector outputs): https://arxiv.org/pdf/0704.1028.pdf
# Niu et al. (2016) proposes a similar scheme, but with binary crossentropy loss on each output node.
# ^paper: http://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html
# Baccianella et al. (2009) propose the "macro" MAE as a metric for OrdReg on unbalanced datasets: https://ieeexplore.ieee.org/abstract/document/5364825   

# Not implemented but still interesting...
# Earth Movers Distance metric https://arxiv.org/pdf/1611.05916.pdf (Hou et al. 2007)
# ROC analyses for OrdReg: https://www.sciencedirect.com/science/article/abs/pii/S0167865507002383 (Waegeman et al. 2008)
# Branco et al. (2016) on imbalanced datasets: https://dl.acm.org/doi/abs/10.1145/2907070
# There is a succinct summary of deep ordinal regression in the introduction here: https://arxiv.org/abs/1705.05278 (2017)

def _cat_to_ranks(y):
    '''
    Converts a batch of one-hot vectors, either the true labels OR the output of a network, to integer rankings.
    [1, 0, 0] --> 1, [.1, .2, .9] --> 3
    '''
    return tf.cast(tf.math.argmax(y, axis=-1), 'float32')


def _ord_probits_to_ranks(y_pred):
    '''
    Interprets a batch of rank-vectors, ouput from a network, as integer rankings.
    [.2, .1, .1] --> 1, [.9, .8, .1] --> 3
    '''
    labels = tf.where(y_pred > 0.5, 1.0, 0.0)
    return _ord_labels_to_ranks(labels)


def _ord_labels_to_ranks(y_true):
    '''
    Interprets a batch of true rank-vector labels as integer rankings.
    [0, 0, 0] --> 1, [1, 1, 0] --> 3
    '''
    return tf.cast(1 + tf.reduce_sum(y_true, axis=-1), 'float32')


def accuracy_k(k, ordi=True):
    '''
    Accuracy within 'k' ranks.
    k=0 is standard accuracy, or "accuracy exact match."
    k=1 is referred to as "accuracy within-one-category-off match" or AEO by Hou et al.
    '''
    (prob2rank, lab2rank) = (_ord_probits_to_ranks, _ord_labels_to_ranks) if ordi else (_cat_to_ranks, _cat_to_ranks)
    def fn(y_true, y_pred):
        pred_rank, true_rank = prob2rank(y_pred), lab2rank(y_true)
        score_bools = tf.abs(pred_rank - true_rank) <= k
        return tf.reduce_mean(tf.cast(score_bools, 'float32'), axis=-1)
    fn.__name__ = 'acc' if k == 0 else f'acc{k}'  # https://stackoverflow.com/questions/57910680/how-to-name-custom-metrics-in-keras-fit-output
    return fn


def mae(ordi=True):
    (prob2rank, lab2rank) = (_ord_probits_to_ranks, _ord_labels_to_ranks) if ordi else (_cat_to_ranks, _cat_to_ranks)
    def fn(y_true, y_pred):
        pred_rank, true_rank = prob2rank(y_pred), lab2rank(y_true)
        dist_abs = tf.abs(pred_rank - true_rank)
        return tf.reduce_mean(dist_abs, axis=-1)
    fn.__name__ = 'mae'
    return fn


def mse(ordi=True):
    (prob2rank, lab2rank) = (_ord_probits_to_ranks, _ord_labels_to_ranks) if ordi else (_cat_to_ranks, _cat_to_ranks)
    def fn(y_true, y_pred):
        pred_rank, true_rank = prob2rank(y_pred), lab2rank(y_true)
        dist_sqr = (pred_rank - true_rank)**2
        return tf.reduce_mean(dist_sqr, axis=-1)
    fn.__name__ = 'mse'
    return fn


def rmse(ordi=True):
    (prob2rank, lab2rank) = (_ord_probits_to_ranks, _ord_labels_to_ranks) if ordi else (_cat_to_ranks, _cat_to_ranks)
    mse_fn = mse(ordi)
    def fn(y_true, y_pred):
        return tf.math.sqrt(mse_fn(y_true, y_pred))
    fn.__name__ = 'rmse'
    return fn

# TODO fix for categorical
class MacroMAE(tf.keras.metrics.Metric):
    def __init__(self, n_ranks, name='mae_macro', ordi=True, **kwargs):
        super(MacroMAE, self).__init__(name=name, **kwargs)
        self.n_ranks = n_ranks
        self.prob2rank = _ord_probits_to_ranks if ordi else _cat_to_ranks
        self.lab2rank = _ord_labels_to_ranks if ordi else _cat_to_ranks
        self.mae_per_rank = [self.add_weight(name=f'mae{i+1}_sum', initializer='zeros') for i in range(n_ranks)]
        self.counts_per_rank = [self.add_weight(name=f'counts{i+1}', initializer='zeros') for i in range(n_ranks)]
        self.final_counts = self.add_weight(name='final_counts', shape=(n_ranks), initializer='zeros')
        self.final_maes = self.add_weight(name='final_maes', shape=(n_ranks), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_rank, true_rank = self.prob2rank(y_pred), self.lab2rank(y_true)
        dist_abs = tf.abs(pred_rank - true_rank)
        for i in range(self.n_ranks):
            label_mask = tf.where(true_rank == i+1, 1.0, 0.0)
            maes = dist_abs * label_mask
            self.mae_per_rank[i].assign_add(tf.reduce_sum(maes, axis=-1))
            self.counts_per_rank[i].assign_add(tf.reduce_sum(label_mask, axis=-1))

    def result(self):
        self.final_counts.assign_add(self.counts_per_rank)
        self.final_maes.assign_add(self.mae_per_rank)
        weighted_maes = tf.math.divide_no_nan(self.final_maes, self.final_counts)
        nonempty_maes = tf.boolean_mask(weighted_maes, tf.where(self.final_counts > 0, True, False))
        return tf.reduce_mean(nonempty_maes)

    def reset_states(self):
        K.batch_set_value([(v, np.zeros(v.shape)) for v in self.variables])