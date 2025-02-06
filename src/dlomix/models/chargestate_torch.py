import warnings
from collections import OrderedDict

import torch.nn as nn

from ..constants import ALPHABET_UNMOD
from ..layers.attention import AttentionLayer

"""
This module contains a model coming in three flavours of predicting precursor charge states for peptide sequences in mass spectrometry data.
The three flavours are:

1. Dominant Charge State Prediction:
   - Task: Predict the dominant charge state of a given peptide sequence.
   - Model: Uses a deep learning model (RNN-based) inspired by Prosit's architecture to predict the most likely charge state.

2. Observed Charge State Prediction:
   - Task: Predict the observed charge states for a given peptide sequence.
   - Model: Uses a multi-label classification approach to predict all possible charge states.

3. Relative Charge State Prediction:
   - Task: Predict the proportion of each charge state for a given peptide sequence.
   - Model: Uses a regression approach to predict the proportion of each charge state.
"""


class ChargeStatePredictorTorch(nn.Module):
    """
    Precursor Charge State Prediction Model for predicting either:
    * the dominant charge state or
    * all observed charge states or
    * the relative charge state distribution
    of a peptide sequence.

    Args:
        embedding_output_dim (int): The size of the embedding output dimension. Defaults to 16.
        seq_length (int): The length of the input sequence. Defaults to 30.
        alphabet (dict): Dictionary mapping for the alphabet (the amino acids in this case). Defaults to ALPHABET_UNMOD.
        dropout_rate (float): The dropout rate used in the encoder layers. Defaults to 0.5.
        latent_dropout_rate (float): The dropout rate for the latent space. Defaults to 0.1.
        recurrent_layers_sizes (tuple): The sizes of the recurrent layers. Defaults to (256, 512).
        regressor_layer_size (int): The size of the regressor layer. Defaults to 512.
        num_classes (int): The number of classes for the output corresponding to charge states available in the data. Defaults to 6.
        model_flavour (str): The type of precursor charge state prediction to be done.
            Can be either "dominant" (using softmax activation), "observed" (using sigmoid activation) or "relative" (using softmax activation).
            Defaults to "relative".
    """

    def __init__(
        self,
        embedding_output_dim=16,
        seq_length=30,
        alphabet=ALPHABET_UNMOD,
        dropout_rate=0.5,
        latent_dropout_rate=0.1,
        recurrent_layers_sizes=(256, 512),
        regressor_layer_size=512,
        num_classes=6,
        model_flavour="relative",
    ):
        super(ChargeStatePredictorTorch, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(alphabet)

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes

        if model_flavour == "relative":
            self.final_activation = nn.Softmax  # Florian also used or "linear"
        elif model_flavour == "observed":
            self.final_activation = nn.Sigmoid
        elif model_flavour == "dominant":
            self.final_activation = nn.Softmax
        else:
            warnings.warn(f"{model_flavour} not available")
            exit

        self.embedding = nn.Embedding(
            num_embeddings=self.embeddings_count,
            embedding_dim=embedding_output_dim,
            padding_idx=0,  # TODO check this
        )

        # build encoder
        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "bidirectional_GRU",
                        nn.GRU(  # TODO check the parameter values
                            input_size=embedding_output_dim,
                            hidden_size=self.recurrent_layers_sizes[0],
                            batch_first=True,
                            bidirectional=True,
                        ),
                    ),
                    ("encoder_dropout1", nn.Dropout(self.dropout_rate)),
                    (
                        "unidirectional_GRU",
                        nn.GRU(  # TODO check the parameter values
                            input_size=recurrent_layers_sizes[0] * 2,
                            hidden_size=self.recurrent_layers_sizes[1],
                            batch_first=True,
                            bidirectional=False,
                        ),
                    ),
                    ("encoder_dropout2", nn.Dropout(self.dropout_rate)),
                ]
            )
        )

        # instead of building a pytorch equivalent for the self-made tf AttentionLayer, use the build-in pytorch MultiheadAttention
        self.attention = nn.MultiheadAttention(  # TODO check the parameter values
            embed_dim=self.recurrent_layers_sizes[1],
            num_heads=1,  # TODO think about increasing this
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense",
                        nn.Linear(
                            in_features=self.recurrent_layers_sizes[1],
                            out_features=self.regressor_layer_size,
                        ),
                    ),
                    ("activation_relu", nn.ReLU()),
                    ("regressor_dropout", nn.Dropout(self.latent_dropout_rate)),
                ]
            )
        )

        self.output_layer = nn.Linear(
            in_features=self.regressor_layer_size, out_features=num_classes
        )

        self.activation = self.final_activation

    def forward(self, inputs):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tensor
            Input tensor (shape: [batch_size, seq_length]).

        Returns
        -------
        tensor
            Predicted output (shape: [batch_size, num_classes]).
        """
        x = self.embedding(inputs)
        x, _ = self.encoder(x)
        x = self.attention(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        x = self.activation(x)
        return x
