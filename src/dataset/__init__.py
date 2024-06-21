import os
from typing import Optional

import gdown

from src.dataset.ds1000 import DS1000Dataset
from src.dataset.stack_overflow import StackOverflowDataset

# Change to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Dataset:
    def __init__(self, dataset: str, kwargs: Optional[dict]) -> None:
        if dataset == "ds-1000":
            if not os.path.exists("ds1000_data"):
                # Download dataset
                gdown.download(
                    id="1sR0Bl4pVHCe9UltBVyhloE8Reztn72VD",
                    output="ds-1000.zip",
                    quiet=False,
                )
                os.system("unzip ds-1000.zip")
                os.system("rm ds-1000.zip")
            self.dataset = DS1000Dataset(source_dir="ds1000_data", mode="Completion")
        elif dataset == "stack_overflow":
            self.dataset = StackOverflowDataset(
                precomputed=kwargs.get("precomputed", False)
            )
        else:
            raise ValueError("Invalid dataset name")
