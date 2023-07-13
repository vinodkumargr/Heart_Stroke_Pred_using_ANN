from src.logger import logging
from src.exception import CustomException
import os, sys
from sklearn.model_selection import train_test_split
from src import entity


class TrainingPipelineConfig:
    def __init__(self):
        try:

            self.artifacts_dir=os.path.join(os.getcwd(), "artifacts")
            os.makedirs(self.artifacts_dir, exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)
        
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifacts_dir,"Data_Ingestion")
            os.makedirs(self.data_ingestion_dir, exist_ok=True)

            self.train_data_path = os.path.join(self.data_ingestion_dir,entity.INGESTION_TRAIN_DATA_FILE)
            self.test_data_path = os.path.join(self.data_ingestion_dir, entity.INGESTION_TEST_DATA_FILE)
            self.validation_data_path = os.path.join(self.data_ingestion_dir, entity.INGESTION_VALIDATION_DATA_FILE)

        except Exception as e:
            raise CustomException(e, sys)
        

class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:

            self.data_validation_dir = os.path.join(training_pipeline_config.artifacts_dir, "Data_Validation")
            os.makedirs(self.data_validation_dir, exist_ok=True)

            self.valid_train_path = os.path.join(self.data_validation_dir, entity.VALID_TRAIN_DATA_FILE)
            self.valid_test_path = os.path.join(self.data_validation_dir, entity.VALID_TEST_DATA_FILE)
            self.valid_validation_path = os.path.join(self.data_validation_dir, entity.VALID_VALIDATION_DATA_FILE)

        except Exception as e:
            raise CustomException(e, sys)
        

class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:

            self.data_transformation_dir = os.path.join(training_pipeline_config.artifacts_dir, "Data_transformation")
            os.makedirs(self.data_transformation_dir, exist_ok=True)

            data_transform_file_path = os.path.join(self.data_transformation_dir, "Data_files")
            os.makedirs(data_transform_file_path, exist_ok=True)

            data_transform_object_path = os.path.join(self.data_transformation_dir, "Object_files")
            os.makedirs(data_transform_object_path, exist_ok=True)

            self.transform_train_path = os.path.join(data_transform_file_path, entity.TRANSFORMATION_TRAIN_DATA_FILE)
            self.transform_validation_path = os.path.join(data_transform_file_path, entity.TRANSFORMATION_VALIDATION_DATA_FILE)
            self.transform_test_path = os.path.join(data_transform_file_path, entity.TRANSFORMATION_TEST_DATA_FILE)
            self.pre_process_object_path = os.path.join(data_transform_object_path, entity.TRANSFORMER_OBJ_FILE)
            self.data_obj_path = os.path.join(data_transform_object_path, entity.DATA_OBJ_FILE)               

        except Exception as e:
            raise CustomException(e, sys)
        

