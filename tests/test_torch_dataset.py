import logging
from os.path import join

import torch
from datasets import Dataset

from dlomix.data import FragmentIonIntensityDataset

logger = logging.getLogger(__name__)

RT_HUB_DATASET_NAME = "Wilhelmlab/prospect-ptms-irt"

RAW_GENERIC_NESTED_DATA = {
    "seq": ["[UNIMOD:737]-DASAQTTSHELTIPN-[]", "[UNIMOD:737]-DLHTGRLC[UNIMOD:4]-[]"],
    "nested_feature": [[[30, 64]], [[25, 35]]],
    "label": [0.1, 0.2],
}


DOWNLOAD_PATH_FOR_ASSETS = join("tests", "assets")


def test_dataset_torch():
    hfdata = Dataset.from_dict(RAW_GENERIC_NESTED_DATA)

    intensity_dataset = FragmentIonIntensityDataset(
        data_format="hf",
        data_source=hfdata,
        sequence_column="seq",
        label_column="label",
        model_features=["nested_feature"],
        dataset_type="pt",
        batch_size=2,
        max_seq_len=15,
    )

    assert intensity_dataset.hf_dataset is not None
    assert intensity_dataset._empty_dataset_mode is False

    batch = next(iter(intensity_dataset.tensor_train_data))

    assert list(batch["nested_feature"].shape) == [1, 1, 2]
    assert list(batch["seq"].shape) == [1, 15]
    assert list(batch["label"].shape) == [
        1,
    ]

    assert batch["seq"].dtype == torch.int64
    assert batch["label"].dtype == torch.float32
