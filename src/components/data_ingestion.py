from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os, sys, json
from src.entity import artifacts_entity, config_entity
from src import utils
from src.constants import database
from sklearn.model_selection import train_test_split
from src import entity


class DataIngestion:
    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomException
        

        
    def initiate_data_ingestion(self) -> artifacts_entity.DataIngestionArtifact:
        try:

            logging.info("Data Ingestion Started .............")

            df = pd.read_csv("/home/vinod/projects_1/Heart_stroke_ANN_/Data/healthcare-dataset-stroke-data.csv")
            df = df.copy()

            #df = pd.DataFrame= utils.get_as_df(database_name=database.DATABASE_NAME, 
            #                                collection_name=database.COLLECTION_NAME)
            logging.info("Read data from MongoDB")


            # writing column names into schema.yaml file
            logging.info("column names writing into schema.yaml file")
            data_columns = df.columns
            report = json.dumps(pd.DataFrame(data_columns).to_dict())
            #report = data_columns.json()
            json_report = json.loads(report)
        
            utils.write_yaml_file(file_path= entity.SCHEMA_YAML_FILE, content=json_report)

            logging.info("written column names into schema.yaml file")

            train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)
            validation_df, test_df = train_test_split(validation_df, test_size=0.5, random_state=42)

            logging.info("spliting into train, test and validation is done")

            logging.info("storing data into artifacts")
            train_df.to_csv(path_or_buf = self.data_ingestion_config.train_data_path, index=False, header=True)
            validation_df.to_csv(path_or_buf = self.data_ingestion_config.validation_data_path, index=False, header=True)
            test_df.to_csv(path_or_buf = self.data_ingestion_config.test_data_path, index=False, header=True)
            
            
            logging.info("stored data into artifacs...")
            logging.info("Exiting data ingestion...")

            data_ingestion_artifact = artifacts_entity.DataIngestionArtifact(
                train_data_path=self.data_ingestion_config.train_data_path,
                validation_data_path=self.data_ingestion_config.validation_data_path,
                test_data_path=self.data_ingestion_config.test_data_path
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)