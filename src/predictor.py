from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity import config_entity, artifacts_entity
from src import entity

import os, sys


# to train new model, if new data comes, and store the new model.

class ModelResolver:

    def __init__(self, model_registry:str="saved_models",
                    transformer_dir_name="transformer",
                    model_dir_name="model"):
        
        try:
            self.model_registry = model_registry
            os.makedirs(self.model_registry, exist_ok=True)
            self.transformer_dir_name = transformer_dir_name
            self.model_dir_name = model_dir_name

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_latest_dir_path(self):
        try:
            
            dir_name=os.listdir(self.model_registry)
            if len(dir_name)==0:
                return None
            
            dir_name = list(map(int, dir_name))
            latest_dir_name=max(dir_name)
            
            return os.path.join(self.model_registry, f"{latest_dir_name}")

        except Exception as e:
            raise CustomException(e, sys)
        

        
    def get_latest_model_path(self):
        try:
            
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("model is not available....")
            
            return os.path.join(latest_dir, self.model_dir_name, entity.MODEL_FILE)

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_latest_transformer_path(self):
        try:
            latest_dir=self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("transformer is not available....")
            
            return os.path.join(latest_dir, self.transformer_dir_name , entity.TRANSFORMER_OBJ_FILE )

        except Exception as e:
            raise CustomException(e, sys)
        


    def get_latest_save_dir_path(self)-> str:
        try:
            
            latest_dir=self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry,f"{1}")
            
            latest_dir_num=int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num}")

        except Exception as e:
            raise CustomException(e, sys)


    def get_latest_save_model_path(self):
        try:
            
            latest_dir=self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, entity.MODEL_FILE)

        except Exception as e:
            raise CustomException(e, sys)
        

    def get_latest_save_transform_path(self):
        try:

            latest_dir=self.get_latest_save_dir_path()
            if latest_dir is None:
                raise ValueError("No saved model directory found.")
            
            return os.path.join(latest_dir, self.transformer_dir_name, entity.TRANSFORMER_OBJ_FILE)

        except Exception as e:
            raise CustomException(e, sys)