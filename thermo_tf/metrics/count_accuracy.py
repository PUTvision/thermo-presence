import tensorflow as tf


class CountAccuracy(tf.keras.metrics.Accuracy):
    def __init__(self, name: str = 'count_accuracy', dtype: tf.dtypes.DType = None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.reduce_sum(y_true, (1, 2)) / 51.35
        y_pred = tf.math.reduce_sum(y_pred, (1, 2)) / 51.35

        y_true = tf.math.round(y_true)
        y_pred = tf.math.round(y_pred)
        
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)
