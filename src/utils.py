import pandas as pd
import numpy as np
import os, sys
from src.exception import CustomException
from src.logger import logging
from src import config
import yaml, pickle
from src.constants import database
from data_dump import client

def get_as_df(database_name:str , collection_name:str)-> pd.DataFrame:
    try:
        logging.info("Reading data from mongoDB")
        df = pd.DataFrame(list(client[database.DATABASE_NAME][database.COLLECTION_NAME].find()))
        
        logging.info(f"Found data shape : {df.shape}")
        if "_id" in df.columns:
            df.drop(['_id'], axis=1, inplace=True)
            logging.info("Found _id and dropped")
        logging.info(f"data shape : {df.shape}")

        return df
    
    except Exception as e:
        raise CustomException(e,sys)


def save_numpy_array_data(file_path: str, array: np.array):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy_array_data(file_path: str):
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)

def append_yaml_file(file_path: str, content: object) -> None:
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"file path : {file_path} not exists...")
        with open(file_path , "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    