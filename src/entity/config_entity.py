from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_file: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    raw_data_file: Path
    train_data_file: Path
    test_data_file: Path

@dataclass(frozen=True)
class ModelBuildingConfig:
    model_params: Dict[str, Optional[Any]]