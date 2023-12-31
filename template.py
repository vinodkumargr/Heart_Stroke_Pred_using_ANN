import os
from pathlib import Path
import logging


logging.basicConfig(
    level = logging.INFO,
    format = "[%(asctime)s : %(levelname)s] : %(message)s"
)


while True:
    project_name = input("Enter the project name : ")
    if project_name != "":
        break
    

logging.info("creating project by name : {project_name}")


files_list = {
    ".github/workflows/.gitkeep",
    ".github/workflows/main.yaml",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/artifacts_entity.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/configuration/mongodb_connection.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/batch_prediction.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/data_access/data.py",
    f"{project_name}/constants/database.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/config.py",
    f"{project_name}/predictor.py",
    f"{project_name}/utils.py",
    "configs/schema.yaml",
    "configs/model.yaml",
    ".gitignore",
    ".env",
    "requirements.txt",
    "setup.py",
    "main.py",
    "data_dump.py"
}


for filepath in files_list:
    filepath = Path(filepath)
    file_dirs , file_names = os.path.split(filepath)
    if file_dirs !="":
        os.makedirs(file_dirs, exist_ok=True)
        logging.info(f"Creating a new directory : {file_dirs} is done......")
    if (not os.path.exists(filepath)):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating a file, called: {file_names} in folder: {file_dirs} is done..... ")
    else:
        logging.info(f"Sorry {filepath} already exists.....")
    