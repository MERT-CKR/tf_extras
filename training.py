import tensorflow as tf

def autofit(model:tf.keras.Sequential,
            X_train:tf.Tensor,
            y_train:tf.Tensor,
            **args):
    """
    Automatically expands input dimensions and fits the model.
    No need to manually call tf.expand_dims(X_train, axis=-1).

    Args:
        model: tf.keras.Sequential
        X_train: Train data
        y_train: Labels
        args: remaining tf.keras.Model.fit arguments
    """
    expanded_X = tf.expand_dims(X_train, axis=-1)
    return model.fit(expanded_X, y_train, **args)
