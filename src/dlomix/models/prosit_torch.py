import warnings
import torch
import torch.nn as nn
from ..constants import ALPHABET_UNMOD
# from ..data.processing.feature_extractors import FEATURE_EXTRACTORS_PARAMETERS
from ..layers.attention import AttentionLayer, DecoderAttentionLayer

class PrositIntensityPredictorTorch(nn.Module):
    def init(
            self,
            embedding_output_dim=16,
            seq_length=30,
            len_fion=6,
            alphabet=None,
            dropout_rate=0.2,
            latent_dropout_rate=0.1,
            recurrent_layers_sizes=(256, 512),
            regressor_layer_size=512,
            use_prosit_ptm_features=False,
            input_keys={
        "SEQUENCE_KEY": "modified_sequence",
    },
            meta_data_keys=None,
            with_termini=True,
        ): 
        super(PrositIntensityPredictorTorch, self).__init__()

        self.dropout_rate = dropout_rate
        self.latent_dropout_rate = latent_dropout_rate
        self.regressor_layer_size = regressor_layer_size
        self.recurrent_layers_sizes = recurrent_layers_sizes
        self.embedding_output_dim = embedding_output_dim
        self.seq_length = seq_length
        self.len_fion = len_fion
        self.use_prosit_ptm_features = use_prosit_ptm_features
        self.input_keys = input_keys
        self.meta_data_keys = meta_data_keys

        # maximum number of fragment ions
        self.max_ion = self.seq_length - 1

        # account for encoded termini
        if with_termini:
            self.max_ion = self.max_ion - 2

        if alphabet:
            self.alphabet = alphabet
        else:
            self.alphabet = ALPHABET_UNMOD

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(self.alphabet)

        self.embedding = nn.Embedding(
            input_dim=self.embeddings_count,
            output_dim=self.embedding_output_dim,
            input_length=seq_length,
        )

        self._build_encoders()
        self._build_decoder()

        self.attention = AttentionLayer(name="encoder_att")

        self.meta_data_fusion_layer = None
        if self.meta_data_keys:
            self.meta_data_fusion_layer = nn.Sequential(
                [
                    torch.mul(name="add_meta"),
                    nn.Tensor.repeat(self.max_ion, name="repeat"),
                ]
            )
        
        self.regressor = nn.Sequential(
            [
                TimeDistributed(
                    nn.Linear(self.len_fion), name="time_dense"   # no specified activation, hence no activation is applied in tf 
                ),
                nn.LeakyReLU(name="activation"),
                nn.Flatten(name="out"),
            ]
        )


    def _build_encoders(self):
        # sequence encoder -> always present
        self.sequence_encoder = nn.Sequential(
            [
                # Need to inplement bidirectionallity of the tf.keras.Bidirectional 
                # bidirectional=True

                #
                # print shapes before and after 
                
                tf.keras.Bidirectional(
                    nn.GRU(
                        units=self.recurrent_layers_sizes[0], return_sequences=True
                    )
                ),
                nn.Dropout(rate=self.dropout_rate),
                nn.GRU(
                    units=self.recurrent_layers_sizes[1], return_sequences=True
                ),
                nn.Dropout(rate=self.dropout_rate),
            
            ]
        )

        # meta data encoder -> optional, only if meta data keys are provided
        self.meta_encoder = None
        if self.meta_data_keys:
            self.meta_encoder = nn.Sequential(
                [
                    torch.cat(name="meta_in"),
                    nn.Linear(
                        self.recurrent_layers_sizes[1], name="meta_dense"
                    ),
                    nn.Dropout(self.dropout_rate, name="meta_dense_do"),
                ]
            )

        # ptm encoder -> optional, only if ptm flag is provided
        self.ptm_input_encoder, self.ptm_aa_fusion = None, None
        if self.use_prosit_ptm_features:
            self.ptm_input_encoder = nn.Sequential(
                [
                    torch.cat(name="ptm_features_concat"),
                    nn.Linear(self.regressor_layer_size // 2),
                    nn.Dropout(rate=self.dropout_rate),
                    nn.Linear(self.embedding_output_dim * 4),
                    nn.Dropout(rate=self.dropout_rate),
                    nn.Linear(self.embedding_output_dim),
                    nn.Dropout(rate=self.dropout_rate),
                ],
                name="ptm_input_encoder",
            )

            self.ptm_aa_fusion = torch.cat(name="aa_ptm_in")

    def _build_decoder(self):
        self.decoder = nn.Sequential(
            [
                nn.GRU(
                    units=self.regressor_layer_size,
                    return_sequences=True,
                    name="decoder",
                ),
                nn.Dropout(rate=self.dropout_rate),
                DecoderAttentionLayer(self.max_ion),
            ]
        )

    def call(self, inputs, **kwargs):
        encoded_meta = None
        encoded_ptm = None

        if not isinstance(inputs, dict):
            # when inputs has (seq, target), it comes as tuple
            peptides_in = inputs
        else:
            peptides_in = inputs.get(self.input_keys["SEQUENCE_KEY"])

            # read meta data from the input dict
            # note that the value here is the key to use in the inputs dict passed from the dataset
            meta_data = self._collect_values_from_inputs_if_exists(
                inputs, self.meta_data_keys
            )

            if self.meta_encoder and len(meta_data) > 0:
                encoded_meta = self.meta_encoder(meta_data)
            else:
                raise ValueError(
                    f"Following metadata keys were specified when creating the model: {self.meta_data_keys}, but the corresponding values do not exist in the input. The actual input passed to the model contains the following keys: {list(inputs.keys())}"
                )

            # read PTM features from the input dict
            ptm_ac_features = self._collect_values_from_inputs_if_exists(
                inputs, PrositIntensityPredictorTorch.PTM_INPUT_KEYS
            )

            if self.ptm_input_encoder and len(ptm_ac_features) > 0:
                encoded_ptm = self.ptm_input_encoder(ptm_ac_features)
            elif self.use_prosit_ptm_features:
                warnings.warn(
                    f"PTM features enabled and following PTM features are expected in the model for Prosit Intesity: {PrositIntensityPredictor.PTM_INPUT_KEYS}. The actual input passed to the model contains the following keys: {list(inputs.keys())}. Falling back to no PTM features."
                )

        x = self.embedding(peptides_in)

        print(x.shape)
        breakpoint()

        # fusion of PTMs (before going into the GRU sequence encoder)
        if self.ptm_aa_fusion and encoded_ptm is not None:
            x = self.ptm_aa_fusion([x, encoded_ptm])

        x = self.sequence_encoder(x)

        x = self.attention(x)

        if self.meta_data_fusion_layer and encoded_meta is not None:
            x = self.meta_data_fusion_layer([x, encoded_meta])
        else:
            # no metadata -> add a dimension to comply with the shape
            x = nn.unsqueeze(x, axis=1)

        x = self.decoder(x)
        x = self.regressor(x)

        return x

    def _collect_values_from_inputs_if_exists(self, inputs, keys_mapping):
        collected_values = []

        keys = []
        if isinstance(keys_mapping, dict):
            keys = keys_mapping.values()

        elif isinstance(keys_mapping, list):
            keys = keys_mapping

        for key_in_inputs in keys:
            # get the input under the specified key if exists
            single_input = inputs.get(key_in_inputs, None)
            if single_input is not None:
                if single_input.ndim == 1:
                    single_input = nn.unsqueeze(single_input, axis=-1)
                collected_values.append(single_input)
        return collected_values
    

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
