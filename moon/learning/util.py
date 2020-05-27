import tensorflow as tf

# Metrics and other utilities for ordinal regression and classification. For OrdReg, a primary
# assumption is that rank1 is represented by [0, 0, ...], rank2 by [1, 0, ...], and so on. For
# classification, a primary assumption is each rank is represented with a one-hot.

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
    return tf.math.argmax(y, axis=-1)


def _ord_probits_to_ranks(y_pred):
    labels = tf.where(y_pred > 0.5, 1.0, 0.0)
    return _ord_labels_to_ranks(labels)


def _ord_labels_to_ranks(y_true):
    return tf.reduce_sum(y_true, axis=-1)

# Accuracy within 'k' ranks.
# k=0 is standard accuracy, or "accuracy exact match."
# k=1 is referred to as "accuracy within-one-category-off match" or AEO by Hou et al.
def accuracy_k(k, ordi=True):
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

# TODO macro-MAE