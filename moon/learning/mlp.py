import tensorflow as tf
from tensorflow.keras import layers

from moon.learning import coral, util

def build_keras_mlp(input_shape, hidden_dim, hidden_layers, hidden_activation, dropout_p, adam_lr, output_type, output_len, y_train=None):
    # Common MLP layers: input and hidden(s)
    in_x = layers.Input(shape=input_shape, name='input')
    features = layers.Flatten()(in_x)
    for i in range(hidden_layers):
        features = layers.Dense(hidden_dim, activation=hidden_activation, name=f'hidden_{i+1}')(features)
        if dropout_p > 0: features = layers.Dropout(dropout_p)(features)
    
    # Different forms for output type, with accompanying loss and metrics
    if 'ordinal_coral' in output_type:
        # CORAL à la Cao et. al (2019)
        out = coral.CoralOutput(output_len)(features)
        logits = layers.Lambda(lambda x: x, name='logits')(out[0])
        probits = layers.Lambda(lambda x: x, name='probits')(out[1])
        loss = {'logits': coral.coral_loss(y_train)}
        metrics = {'probits': [util.MacroMAE(n_ranks=output_len+1), util.accuracy_k(1), util.mae(), util.accuracy_k(0)]}
        name = 'MLP_Coral'
    elif output_type == 'ordinal_vanilla':
        # Vanilla in the vein of Cheng et. al (2007), with k-1 independent sigmoid outputs with MSE loss
        logits = layers.Dense(output_len, name='logits')(features)
        probits = layers.Activation('sigmoid', name='probits')(logits)
        loss = {'probits': 'mse'}
        metrics = {'probits': [util.MacroMAE(n_ranks=output_len+1), util.accuracy_k(1), util.mae(), util.accuracy_k(0)]}
        name = 'MLP_OrdReg_Vanilla'
    elif 'categorical' in output_type:
        # Standard classifier
        logits = layers.Dense(output_len, name='logits')(features)
        probits = layers.Softmax(name='probits')(logits)
        loss = {'logits': tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=(0.1 if output_type == 'categorical_smoothed' else 0))}
        metrics = {'probits': [util.MacroMAE(n_ranks=output_len), util.accuracy_k(1, ordi=False), util.mae(ordi=False), util.accuracy_k(0, ordi=False)]}
        name = 'MLP_Categorical'
    else:
        raise NotImplementedError(f'Output type "{output_type}" not recognized')
    
    optim = tf.keras.optimizers.Adam(lr=adam_lr)
    model = tf.keras.Model(in_x, [logits, probits], name=name)
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    # model.summary()
    return model