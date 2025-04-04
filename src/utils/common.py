import yaml
import os 
from pathlib import Path
from ensure import ensure_annotations
from src.utils.logger import logger

@ensure_annotations
def read_yaml(file_path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")