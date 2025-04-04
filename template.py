import os
from pathlib import Path
import logging

os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename='logs/project_setup.log',
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



list_of_files = [
   # "artifacts/",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "config/config.yaml",
    "Dockerfile",
    "params.yaml",
    "dvc.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "notebooks/research.ipynb"
    #"templates/",
    #".github/workflows/cicd.yaml"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !='':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory; {filedir} for the file: {filename}')
    

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w' ) as f:
            pass
            logging.info(f'Creating file: {filename}')
    
    else:
        logging.info(f'File already exists: {filename}')



    "artifacts/",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_preprocesing.py",
    "src/components/model_building.py",
    "src/components/model_evaluation.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/utils/exception.py",
    "src/utils/logger.py",
    "src/config/__init__.py",
    "src/config/configuration.py",
    "src/pipeline/__init__.py",
    "src/entity/__init__.py",
    "src/entity/config_entity.py",
    "config/config.yaml",
    "Dockerfile",
    "params.yaml",
    "dvc.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    "notebooks/research.ipynb"