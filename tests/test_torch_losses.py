import logging

import numpy as np
import tensorflow as tf
import torch

from dlomix.losses import masked_spectral_distance
from dlomix.losses import masked_spectral_distance_torch

logger = logging.getLogger(__name__)


# ------------------ intensity - masked spectral distance ------------------


def test_tf_torch_equivalence_masked_spectral_distance():
    
    y_true = [[0.1, 0.2, 0.3]]
    y_pred = [list(reversed(y_true[0]))]

    sa_tf = masked_spectral_distance(
        tf.convert_to_tensor(y_true), 
        tf.convert_to_tensor(y_pred)
    )
    sa_torch = masked_spectral_distance_torch(
        torch.tensor(y_true), 
        torch.tensor(y_pred)
    )

    logger.info(f"Spectral Angle: for tf: {sa_tf.numpy()} vs for torch: {sa_torch.numpy()}")

    assert (sa_tf.numpy() == sa_torch.numpy()).all() # alternatively try np.array_equiv(A,B)