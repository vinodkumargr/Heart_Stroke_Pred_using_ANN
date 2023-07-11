from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.entity import artifacts_entity, config_entity
from src import utils
import pandas as pd
import numpy as np
import os, sys, json


class DataValidation:
    def __init__(self, 
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifacts:artifacts_entity.DataIngestionArtifact):
        
        try:

            self.data_ingestion_artifacts=data_ingestion_artifacts
            self.data_validation_config=data_validation_config

        except Exception as e:
            raise CustomException(e, sys)
        

    