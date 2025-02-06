"""Collaborative Filtering model for imputation.

with Ligthening.
"""

# %%
import pathlib

import lightning as pl
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
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
col_map = {"Sample ID": "sample", "protein_group": "feature"}
data = pd.read_csv(DATASOURCE).rename(col_map, axis=1)
cats = data.select_dtypes(exclude="number").columns
print(f"Selected as categories: {cats}")
data = data.astype({c: "category" for c in cats})
data


# %%
data[cats].describe()

# %%
n_samples, n_features = data[cats].nunique()

# %% [markdown]
# Lookup table for categories: category to integer index
# - used to get embedding

# %%
lookup = {}
for cat in cats:
    lookup[cat] = {c: i for i, c in enumerate(data[cat].cat.categories)}
lookup

# %% [markdown]
# - load tabular dataset (long format of LFQ protein groups)

# %%
# ds = load_dataset(
#     "csv",
#     data_files=DATASOURCE,
#     split="train",
#     cache_dir=None,
# ).with_format("torch", device=device)
# load from pandas for now
ds = dataset = Dataset.from_pandas(data).with_format("torch")

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
    batch["sample"],
    batch["feature"],
    batch["intensity"],
)

# %% [markdown]
# Model
# - coupling to external knowledge about the data (categories and lookup)
# - embedding indices have to be look-up on each forward pass
# - same dimension for sample and feature embedding, simple dot-product loss with MSE


# %%
class CollaborativeFilteringModel(pl.LightningModule):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        lookup: dict[str, dict[str, int]],
        learning_rate: float = 0.001,
        embedding_dim: int = 32,
        device: str = "cpu",
    ):
        super(CollaborativeFilteringModel, self).__init__()
        self.sample_embedding = nn.Embedding(num_samples, embedding_dim, device=device)
        self.feature_embedding = nn.Embedding(
            num_features, embedding_dim, device=device
        )
        self.lookup = lookup
        self.fc = nn.Linear(embedding_dim, 1)
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, sample_ids, feature_ids):
        # lookup integers
        sample_ids = torch.tensor(
            [self.lookup["sample"][sample] for sample in sample_ids]
        )
        feature_ids = torch.tensor(
            [self.lookup["feature"][feat] for feat in feature_ids]
        )
        sample_embeds = self.sample_embedding(sample_ids)
        feature_embeds = self.feature_embedding(feature_ids)
        sample_embeds
        dot_product = (sample_embeds * feature_embeds).sum(1)
        return dot_product

    def training_step(self, batch):
        # sample_ids, feature_ids, intensities = batch
        sample_ids, feature_ids, intensities = (
            batch["sample"],
            batch["feature"],
            batch["intensity"],
        )

        predictions = self(sample_ids, feature_ids)
        loss = self.loss(predictions, intensities.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        # sample_ids, feature_ids, intensities = batch
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
)
model

# %%
trainer = pl.Trainer(accelerator="cpu", max_epochs=2)
tuner = Tuner(trainer)


# %%
# Create a Tuner
tuner.lr_find(model, train_dataloaders=dl_train, attr_name="learning_rate")

# %%
trainer.fit(model, dl_train)
trainer.fit(model, dl_train)
