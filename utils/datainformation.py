from dataclasses import dataclass
from typing import List

import pandas as pd

@dataclass()
class DataInformation:
    df: pd.DataFrame
    features: List[str]
    target: str
    dataset_name: str