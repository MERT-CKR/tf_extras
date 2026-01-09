import tensorflow as tf

def mae(y_true, y_pred) -> tf.Tensor:
    """
    Compute the Mean Absolute Error (MAE) between y_true and y_pred.

    Args:
        y_true (tf.Tensor | array-like): Ground truth values.
        y_pred (tf.Tensor | array-like): Predicted values.

    Returns:
        tf.Tensor: Scalar tensor representing the MAE.

    Notes:
        Automatically casts both inputs to float32 to prevent dtype mismatch errors.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return tf.reduce_mean(tf.abs(y_true - y_pred))

