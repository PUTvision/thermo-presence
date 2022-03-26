import tensorflow as tf


class CountMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, person_point_weight: float, name: str = 'count_mean_absolute_error', dtype: tf.dtypes.DType = None):
        super().__init__(name, dtype)
        self._person_point_weight = person_point_weight


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.reduce_sum(y_true, (1, 2)) / self._person_point_weight
        y_pred = tf.math.reduce_sum(y_pred, (1, 2)) / self._person_point_weight

        y_true = tf.math.round(y_true)
        y_pred = tf.math.round(y_pred)
        
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


    def get_config(self):
        return {"person_point_weight": self._person_point_weight}
