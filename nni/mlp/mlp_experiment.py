import os
import nni
import numpy as np
import tensorflow as tf
import random as random

import moon.data
import moon.ml.coral
import moon.ml.mlp
import moon.ml.util

def run_trial(params):
    os.environ['PYTHONHASHSEED'] = str(params['seed'])
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    tf.random.set_seed(params['seed'])
    probs = moon.data.read_problems('../../data/cleaned_probs.csv')
    splits = moon.data.split_data(probs, seed=params['seed'])
    x_train, x_val, x_test = [np.array([p.array for p in probs]) for probs in splits]
    
    # Label format and output length depend on output_type
    if 'categorical' in params['output_type']:
        y_train, y_val, y_test = [np.array([p.grade.categorical for p in probs]) for probs in splits]
        output_len = moon.problem.Grade.N_GRADES  # y.shape[-1]
    elif 'ordinal' in params['output_type']:
        y_train, y_val, y_test = [np.array([p.grade.ordinal_rank for p in probs]) for probs in splits]
        output_len = moon.problem.Grade.N_GRADES - 1  # y.shape[-1]
    # Problem format depends on flattened_input
    input_shape = moon.problem.Problem.GRID_SHAPE if params['flattened_input'] else (*moon.problem.Problem.GRID_SHAPE, 3)
    # Coral output can have a weighted loss function; it needs y_train to calculate the importance weights
    y_train_for_model = y_train if params['output_type'] == 'ordinal_coral_weighted' else None

    # Build model
    m = moon.ml.mlp.build_keras_mlp(
        input_shape=input_shape,
        hidden_dim=params['hidden_dim'],
        hidden_layers=params['hidden_layers'],
        hidden_activation=params['hidden_activation'],
        dropout_p=params['dropout_p'],
        adam_lr=params['adam_lr'],
        output_type=params['output_type'],
        output_len=output_len,
        y_train=y_train_for_model
    )
    # Train model
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, verbose=1)]  # TODO: tensorboard and intermediate logging to nni https://github.com/microsoft/nni/blob/master/examples/trials/mnist-keras/mnist-keras.py
    batch_size = params['batch_size']
    max_epochs = params['max_epochs']
    # Test model and send to NNI
    history = m.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=max_epochs, callbacks=callbacks)
    test_metrics = m.evaluate(x_test, y_test, return_dict=True, batch_size=len(x_test))
    test_metrics = moon.ml.util.keras_metrics_to_nni(test_metrics)
    print(test_metrics)
    # nni.report_final_result(test_metrics)


if __name__ == '__main__':
    params = dict(
        seed = 21,
        batch_size = 64,
        max_epochs = 5,
        flattened_input = True,
        hidden_dim = 16,
        hidden_layers = 1,
        hidden_activation = 'swish',
        dropout_p = 0.5,
        adam_lr = 1e-3,
        # output_type = 'categorical_unsmoothed',
        # output_type = 'categorical_smoothed',
        output_type = 'ordinal_coral',
    )
    # params = nni.get_next_parameter()
    run_trial(params)