from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import os, sys
from src.entity import artifacts_entity, config_entity
from src import utils
from src.constants import database
from sklearn.model_selection import train_test_split



class DataIngestion:
    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise CustomException
        
        
    def split_into_train_validation(self, df:pd.DataFrame) -> pd.DataFrame:
        try:

            data = pd.read_csv(df)
            X_train, X_val, y_train, y_val = train_test_split(data, test_size=0.2, random_state=42)

            return X_train, y_train
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def split_into_validation_test(self, df:pd.DataFrame)-> pd.DataFrame:
        try:

            data = pd.read_csv(df)
            

        except Exception as e:
            raise CustomException(e, sys)
        

        
    def initiate_data_ingestion(self) -> artifacts_entity.DataIngestionArtifact:
        try:

            logging.info("Data Ingestion Started .............")

            df = pd.read_csv("/home/vinod/projects_1/Heart_stroke_ANN_/Data/healthcare-dataset-stroke-data.csv")
            df = df.copy()

            #df = pd.DataFrame= utils.get_as_df(database_name=database.DATABASE_NAME, 
            #                                collection_name=database.COLLECTION_NAME)
            logging.info("Read data from MongoDB")


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