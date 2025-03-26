import yaml

def read_yaml(file_path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)