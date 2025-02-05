import tensorflow as tf
import tensorflow.keras.backend as K

from dlomix.eval import tf as tf_eval
from dlomix.types import Tensor


# code adopted and modified based on:
# https://github.com/horsepurve/DeepRTplus/blob/cde829ef4bd8b38a216d668cf79757c07133b34b/RTdata_emb.py
def delta95_metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
    if isinstance(y_true, tf.Tensor):
        ret = tf_eval.rt_eval.delta95_metric(y_true, y_pred)
        return ret
    else:
        raise NotImplementedError("todo")


if __name__ == "__main__":
    # test case: absolute error is 2.0 is  below 95th percentile
    y_true = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = tf.constant([1.5, 3.0, 4.5, 6.0, 7.5])
    # abs_error =        [0.5, 1.0, 1.5, 2.0, 2.5]

    print(delta95_metric(y_true, y_pred))  # 4 / 4
