import os
import sys

import pandas as pd
import tensorflow as tf

from dlomix.data import RetentionTimeDataset
from dlomix.eval import TimeDeltaMetric
from dlomix.models import PrositRetentionTimePredictor
from dlomix.reports import RetentionTimeReport

TRAIN_DATAPATH = "example_dataset/proteomTools_train_val.csv"
TEST_DATAPATH = "example_dataset/proteomTools_test.csv"

d = RetentionTimeDataset(
    data_format="csv",
    data_source=TRAIN_DATAPATH,
    test_data_source=TEST_DATAPATH,
    sequence_column="sequence",
    label_column="irt",
    max_seq_len=30,
    batch_size=512,
    val_ratio=0.2,
    dataset_type="pt",
)

print(d)
print(d["train"]["sequence"][0:2])
print(d["train"]["irt"][0:2])

test_targets = d["test"]["irt"]
test_sequences = d["test"]["sequence"]

for x in d.tensor_train_data:
    print(x)
    break


# TODO: Continue with models

print(1 / 0)

model = PrositRetentionTimePredictor(seq_length=30)
optimizer = tf.keras.optimizers.Adam(lr=0.0001)

model.compile(
    optimizer=optimizer, loss="mse", metrics=["mean_absolute_error", TimeDeltaMetric()]
)

weights_file = "./prosit_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
callbacks = [checkpoint, early_stop, decay]


history = model.fit(
    d.tensor_train_data,
    epochs=25,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)

predictions = model.predict(test_sequences)
predictions = predictions.ravel()

print(test_sequences[:5])
print(test_targets[:5])
print(predictions[:5])


report = RetentionTimeReport(output_path="run_scripts/output", history=history)

print("R2: ", report.calculate_r2(test_targets, predictions))

pd.DataFrame(
    {
        "sequence": d["test"]["_parsed_sequence"],
        "irt": test_targets,
        "predicted_irt": predictions,
    }
).to_csv("run_scripts/output/predictions_prosit_fullrun.csv", index=False)
