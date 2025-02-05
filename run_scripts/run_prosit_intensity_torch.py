import os
import sys

import tensorflow as tf

from dlomix.data import FragmentIonIntensityDataset
from dlomix.losses import masked_spectral_distance
from dlomix.models import PrositIntensityPredictorTorch

# consider the use-case for starting from a saved model

model = PrositIntensityPredictorTorch(

)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

TRAIN_DATAPATH = "example_dataset/intensity/third_pool_processed_sample.parquet"

d = FragmentIonIntensityDataset(
    data_source=TRAIN_DATAPATH,
    max_seq_len=30,
    batch_size=8,
    val_ratio=0.2,
    # model_features=["collision_energy_aligned_normed", "precursor_charge_onehot"],
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    features_to_extract=["mod_loss", "delta_mass"],
    dataset_type="pt"
)

# model.compile(optimizer=optimizer, loss=masked_spectral_distance, metrics=["mse"])

weights_file = "run_scripts/output/prosit_intensity_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop, decay]

# make training loop 

model.train()
loss = masked_spectral_distance


for epoch in range(0, 10):
    for batch in d.tensor_train_data:
        optimizer.zero_grad()

        output = PrositIntensityPredictorTorch(model)
        loss = loss(output, )



        

# # to add test data, a pool for example
# td = IntensityDataset(
#     data_source=TRAIN_DATAPATH,
#     seq_length=30,
#     batch_size=128,
#     val_ratio=0,
#     precursor_charge_col="precursor_charge_onehot",
#     sequence_col="modified_sequence",
#     collision_energy_col="collision_energy_aligned_normed",
#     intensities_col="intensities_raw",
#     features_to_extract=[
#         ModificationLocationFeature(),
#         ModificationLossFeature(),
#         ModificationGainFeature(),
#     ],
#     parser="proforma",
#     test=True,
# )
# predictions = model.predict(td.test_data)

# print(predictions.shape)
# print(predictions[0])

# from dlomix.reports import IntensityReport

# # create a report object by passing the history object and plot different metrics
# report = IntensityReport(output_path="./output", history=history)
# report.generate_report(td, predictions)
# # you can also manually see the results by calling other utility functions

# from dlomix.reports.postprocessing import normalize_intensity_predictions

# predictions_df = report.generate_intensity_results_df(td, predictions)
# predictions_df.to_csv("./predictions_prosit_intensity_ptm_fullrun.csv", index=False)
