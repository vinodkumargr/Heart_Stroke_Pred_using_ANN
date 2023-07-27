from src.exception import CustomException
import os, sys
import pandas as pd
import numpy as np
from src.predictor import ModelResolver
from src import utils

PREDICTION_DIR = "Prediction"

def strat_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        model_resolver = ModelResolver(model_registry="saved_models")

        # Load data:
        data = pd.read_csv(input_file_path)
        data = data.dropna(axis=0)

        # Load transformer
        transformer = utils.load_object(file_path=model_resolver.get_latest_save_transform_path())

        # Transform the data
        input_arr = transformer.transform(data)

        # Load the model
        model = utils.load_object(file_path=model_resolver.get_latest_save_model_path())

        # Predict probabilities from the data using the model
        probabilities = model.predict(input_arr)

        # Apply thresholding to convert probabilities to class labels (0 or 1)
        threshold = 0.5
        predictions = (probabilities > threshold).astype(int)

        # Create a column for new predictions
        data['prediction'] = predictions

        # Save this prediction data as a new dataframe
        prediction_file_name = os.path.join(PREDICTION_DIR, "prediction_file.csv")

        # Convert the data into CSV
        data.to_csv(path_or_buf=prediction_file_name, index=False, header=True)

        return prediction_file_name

    except Exception as e:
        raise CustomException(e, sys)
