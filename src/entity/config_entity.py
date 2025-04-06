from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_file: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: str
    raw_data_file: Path
    train_data_file: Path
    test_data_file: Path
    transformer_name: str

@dataclass(frozen=True)
class ModelBuildingConfig:
      model_dir: str
      model_params: Dict[str, Optional[Any]]
      model_name: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
      model_dir: Path
      metrics_dir: str
      metrics_name: str

@dataclass(frozen=True)
class MlflowConfig:
    metrics_dir: Path
    model_dir: Path
    model_name: str
    tracking: Dict[str, Any]

@dataclass(frozen=True)
class DriftDetectionConfig:
    drift_dir: str
    train_data_file: Path
    test_data_file: Path
    drift_name: str