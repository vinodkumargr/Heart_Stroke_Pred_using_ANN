import pymongo
import pandas as pd
import json
from src.constants import database
import os
from dotenv import load_dotenv


load_dotenv()
client = pymongo.MongoClient(os.getenv("MONGO_DB_URL"))


DATA_FILE_PATH = "/home/vinod/projects_1/Heart_stroke_ANN_/Data/healthcare-dataset-stroke-data.csv"
DATABASE_NAME = database.DATABASE_NAME
COLLECTION_NAME = database.COLLECTION_NAME

if __name__ == "__main__":
    data = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and columns : {data.shape}")
    
    
    data.drop(columns="id", inplace=True)
    
    
    json_record = list(json.loads(data.T.to_json()).values())
    print(json_record[0])
    
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)