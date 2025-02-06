"""Collaborative Filtering model for imputation.

with Ligthening: `pip install lightning`
"""

# %%
import pathlib

import lightning as pl
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from lightning.pytorch.tuner import Tuner
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.mps.is_available() else "cpu")
device

# %% [markdown]
# Load data from URL once

# %%
DATASOURCE = pathlib.Path("protein_groups_wide_N50_M454.csv")
if not DATASOURCE.exists():
    URLSOURCE = (
        "https://raw.githubusercontent.com/RasmussenLab/pimms/refs/heads/main/"
        "project/data/dev_datasets/HeLa_6070/protein_groups_wide_N50_M454.csv"
    )
    data = (
        pd.read_csv(URLSOURCE, index_col=0)
        .rename_axis("protein_group", axis=1)
        .stack()
        .squeeze()
        .to_frame("intensity")
    )
    data.to_csv("protein_groups_wide_N50_M454.csv")  # local dump

# %% [markdown]
# Load data from disk

# %%
col_map = {"sample": "Sample ID", "feature": "protein_group", "intensity": "intensity"}

# %% [markdown]
# - load tabular dataset (long format of LFQ protein groups)

# %%
ds = load_dataset(
    "csv",
    data_files=str(DATASOURCE),
    split="train",
    cache_dir=None,
).with_format("torch")
ds

# %%
set_samples = set(ds[col_map["sample"]])
n_samples = len(set_samples)
set_features = set(ds[col_map["feature"]])
n_features = len(set_features)

lookup = dict()
lookup["sample"] = {c: i for i, c in enumerate(set_samples)}
lookup["feature"] = {c: i for i, c in enumerate(set_features)}

# %%
ds_dict = ds.train_test_split(test_size=0.2)
ds_dict

# %% [markdown]
# inspect a single dataset entry and a batch

# %%
for item in ds_dict["train"]:
    break
item

# %%
dl_train = DataLoader(ds_dict["train"], batch_size=256)
for batch in dl_train:
    break
batch

# %%
sample_ids, feature_ids, intensities = (
    batch[col_map["sample"]],
    batch[col_map["feature"]],
    batch[col_map["intensity"]],
)

# %% [markdown]
# Model
# - coupling to external knowledge about the data (categories and lookup)
# - embedding indices have to be look-up on each forward pass
# - same dimension for sample and feature embedding, simple dot-product loss with MSE


# %%
COL_MAP = {"sample": "sample", "feature": "feature", "intensity": "intensity"}


class CollaborativeFilteringModel(pl.LightningModule):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        lookup: dict[str, dict[str, int]],
        col_map: dict[str, str] = COL_MAP,
        learning_rate: float = 0.001,
        embedding_dim: int = 32,
    ):
        super(CollaborativeFilteringModel, self).__init__()
        self.sample_embedding = nn.Embedding(num_samples, embedding_dim, device=device)
        self.feature_embedding = nn.Embedding(
            num_features, embedding_dim, device=device
        )
        self.col_map = col_map
        self.lookup = lookup
        self.fc = nn.Linear(embedding_dim, 1)
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, sample_ids, feature_ids):
        sample_ids = torch.tensor(
            [self.lookup["sample"][sample] for sample in sample_ids],
            device=self.device,
        )
        feature_ids = torch.tensor(
            [self.lookup["feature"][feat] for feat in feature_ids],
            device=self.device,
        )
        sample_embeds = self.sample_embedding(sample_ids)
        feature_embeds = self.feature_embedding(feature_ids)
        sample_embeds
        dot_product = (sample_embeds * feature_embeds).sum(1)
        return dot_product

    def training_step(self, batch):
        sample_ids, feature_ids, intensities = (
            batch[self.col_map["sample"]],
            batch[self.col_map["feature"]],
            batch[self.col_map["intensity"]],
        )

        predictions = self(sample_ids, feature_ids)
        loss = self.loss(predictions, intensities.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        sample_ids, feature_ids, intensities = (
            batch["sample"],
            batch["feature"],
            batch["intensity"],
        )
        predictions = self(sample_ids, feature_ids)
        loss = self.loss(predictions, intensities.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


model = CollaborativeFilteringModel(
    num_samples=n_samples,
    num_features=n_features,
    lookup=lookup,
    col_map=col_map,
)
model

# %%
trainer = pl.Trainer(accelerator=str(device), max_epochs=10)
tuner = Tuner(trainer)

# %%
# Create a Tuner
tuner.lr_find(model, train_dataloaders=dl_train, attr_name="learning_rate")

# %%
trainer.fit(model, dl_train)

# %%
