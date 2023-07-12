from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src import entity
from src.entity import artifacts_entity, config_entity
from src import utils
import pandas as pd
import numpy as np
import os, sys
from src.utils import read_yaml_file, write_yaml_file, append_yaml_file


class DataValidation:
    def __init__(self, 
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifacts:artifacts_entity.DataIngestionArtifact):
        
        try:

            self.data_ingestion_artifacts=data_ingestion_artifacts
            self.data_validation_config=data_validation_config

        except Exception as e:
            raise CustomException(e, sys)
        

    def all_columns_exists(self, previous_df:pd.DataFrame, present_df:pd.DataFrame):
        try:

            status=False
            missing_cols = []
            for column in previous_df.columns:
                if column not in present_df.columns:
                    missing_cols.append(column)

            logging.info(f"Missing columns : {missing_cols}")
            append_yaml_file(file_path=entity.SCHEMA_YAML_FILE, content=missing_cols)

            if len(missing_cols)==0:                
                return True
            else:
                raise CustomException(f"{missing_cols} not detects ")

        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_validation(self) -> artifacts_entity.DataValidationArtifact:
        try:

            logging.info("data validation stage started ....")

            logging.info("reading data from data_ingestion_artifacts")
            train_df = pd.read_csv(self.data_ingestion_artifacts.train_data_path)
            validation_df = pd.read_csv(self.data_ingestion_artifacts.validation_data_path)
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_data_path)

            logging.info("In train_df : ")
            train_status = self.all_columns_exists(previous_df=pd.read_csv(self.data_ingestion_artifacts.train_data_path),
                                                   present_df=train_df)
            
            logging.info("In validation_df : ")
            validation_status = self.all_columns_exists(previous_df=pd.read_csv(self.data_ingestion_artifacts.validation_data_path),
                                                        present_df=validation_df)
            
            logging.info("In test_df : ")
            test_status = self.all_columns_exists(previous_df=pd.read_csv(self.data_ingestion_artifacts.test_data_path),
                                                  present_df=test_df)


            logging.info("Exporting data into data_validation_artifacts")

            train_df.to_csv(path_or_buf=self.data_validation_config.valid_train_path, index=False, header=True)
            validation_df.to_csv(path_or_buf=self.data_validation_config.valid_validation_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_validation_config.valid_test_path, index=False, header=True)


            data_validation_artifacts = artifacts_entity.DataIngestionArtifact(
                train_data_path=self.data_validation_config.valid_train_path,
                validation_data_path=self.data_validation_config.valid_validation_path,
                test_data_path=self.data_validation_config.valid_test_path
            )

            logging.info("Exiting data_validation stage")

            return data_validation_artifacts

        except Exception as e:
            raise CustomException(e, sys)