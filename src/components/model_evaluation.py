from src.exception import CustomException
from src.logger import logging
from src import config, utils
from src.entity import config_entity, artifacts_entity
from src.predictor import ModelResolver
import os, sys, re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from src import entity
from src.utils import read_yaml_file, write_yaml_file, append_yaml_file

class ModelEvaluation:
    def __init__(self, model_evaluation_config: config_entity.ModeEvaluationConfig,
                 data_ingestion_artifacts: artifacts_entity.DataIngestionArtifact,
                 data_validation_artifacts: artifacts_entity.DataValidationArtifact,
                 data_transformation_artifacts: artifacts_entity.DataTransformationArtifact,
                 model_trainer_artifacts: artifacts_entity.ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifacts
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_artifacts = model_trainer_artifacts
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> artifacts_entity.ModelEvaluationArtifact:
        try:
            logging.info("Model evaluation started ......")
            logging.info("getting latest model path")
            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path is None:  # if the model accuracy is not increased, then it will not create new model dirs
                model_evaluation_artifact = artifacts_entity.ModelEvaluationArtifact(
                    model_accepted=False,
                    improved_accuracy=None
                )
                logging.info(f"model_evaluation_artifact : {model_evaluation_artifact}")
                
                return model_evaluation_artifact

            # find previous/old model location
            logging.info("finding old model path...")
            old_transformer_path = self.model_resolver.get_latest_save_transform_path()
            old_model_path = self.model_resolver.get_latest_model_path()

            # read previous model
            logging.info("reading old model...")
            old_transformer = utils.load_object(old_transformer_path)
            old_model = utils.load_object(file_path=old_model_path)

            # read current/new model
            logging.info("reading new model...")
            current_transformer = utils.load_object(self.model_trainer_artifacts.pre_processing_obj)
            current_model = utils.load_object(file_path=self.model_trainer_artifacts.model_path)

            # reading old test data and predicting for old model
            old_train_data = self.data_transformation_artifacts.transform_train_path
            old_train_data = utils.load_numpy_array_data(old_train_data)
            old_x_train, old_y_train = old_train_data[:, :-1], old_train_data[:, -1]

            old_model_y_pred = (old_model.predict(old_x_train) > 0.5).astype(int)

            # previous model F1-score
            logging.info("comparing models....")
            previous_model_f1_score = f1_score(y_true=old_y_train, y_pred=old_model_y_pred)
            logging.info(f"previous model F1-score: {previous_model_f1_score}")

            # reading new test data and predicting for new model
            new_train_data = self.data_transformation_artifacts.transform_train_path
            new_train_data = utils.load_numpy_array_data(new_train_data)
            new_x_train, new_y_train = new_train_data[:, :-1], new_train_data[:, -1]

            new_model_y_pred = (current_model.predict(new_x_train) > 0.5).astype(int)

            current_model_f1_score = f1_score(y_true=new_y_train, y_pred=new_model_y_pred)
            logging.info(f"current model F1-score: {current_model_f1_score}")

            # Choose the better performing model
            best_model, best_model_f1_score = (
                (old_model, previous_model_f1_score)
                if previous_model_f1_score > current_model_f1_score
                else (current_model, current_model_f1_score)
            )


            model_accepted = current_model_f1_score > previous_model_f1_score

            model_evaluation_artifact = artifacts_entity.ModelEvaluationArtifact(
                model_accepted=model_accepted,
                improved_accuracy=best_model_f1_score - previous_model_f1_score,
            )

            logging.info("Exiting model Evaluation stage")

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)
