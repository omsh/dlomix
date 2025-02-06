import tensorflow as tf

from dlomix.constants import PTMS_ALPHABET
from dlomix.data import ChargeStateDataset
from dlomix.models import ChargeStatePredictor

model = ChargeStatePredictor(
    num_classes=6, seq_length=32, alphabet=PTMS_ALPHABET, model_flavour="dominant"
)
print(model)


optimizer = tf.keras.optimizers.Adam(lr=0.0001)

d = ChargeStateDataset(
    data_format="hub",
    data_source="Wilhelmlab/prospect-ptms-charge",
    sequence_column="modified_sequence",
    label_column="most_abundant_charge_state",
    max_seq_len=30,
    batch_size=512,
)
print(d)

for x in d.tensor_train_data:
    print(x)
    break

test_targets = d["test"]["most_abundant_charge_state"]
test_sequences = d["test"]["modified_sequence"]

# callbacks
weights_file = "./output/prosit_charge_major_test"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    weights_file, save_best_only=True, save_weights_only=True
)
early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
decay = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=10, verbose=1, min_lr=0
)
callbacks = [checkpoint, early_stop, decay]


model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)


history = model.fit(
    d.tensor_train_data,
    epochs=2,
    validation_data=d.tensor_val_data,
    callbacks=callbacks,
)

predictions = model.predict(test_sequences)
predictions = predictions.ravel()

print(test_sequences[:5])
print(test_targets[:5])
print(predictions[:5])
