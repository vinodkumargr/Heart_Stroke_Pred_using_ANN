from src.logger import logging
from src.exception import CustomException
from src import entity
from src.entity import artifacts_entity, config_entity
from src import utils
from src.constants.database import TARGET_COLUMN
import pandas as pd
import numpy as np
import os, sys
from src.utils import read_yaml_file, write_yaml_file, append_yaml_file
from imblearn.combine import SMOTEENN
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer


class DataTransformation:

    def __init__(self, data_validation_artifacts:artifacts_entity.DataValidationArtifact,
                        data_transformation_config:config_entity.DataTransformationConfig):
        
        try:      

            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise CustomException(e, sys)
        
        

    def get_transformer(self):
        try:
            
            numerical_columns = ['age', 'hypertension', 'heart_disease']
            categorical_columns = ['ever_married', 'work_type', 'smoking_status']
            transform_columns = ['avg_glucose_level', 'bmi']
        
            numeric_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ]
            )

            transform_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('transformer', PowerTransformer(standardize=True))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("Numeric_Pipeline",numeric_pipeline,numerical_columns),
                    ("Categorical_Pipeline",categorical_pipeline, categorical_columns),
                    ("Power_Transformation", transform_pipe, transform_columns)
            ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info(
                "Exited get_transformer method of DataTransformation class"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

        
    def initiate_data_transformation(self) -> artifacts_entity.DataTransformationArtifact:
        try:

            logging.info("Data Transformation is started ...")

            logging.info("reading data from data_ingestion_artifacts")
            train_df = pd.read_csv(self.data_validation_artifacts.valid_train_path)
            validation_df = pd.read_csv(self.data_validation_artifacts.valid_validation_path)
            test_df = pd.read_csv(self.data_validation_artifacts.valid_test_path)


            #split into input and out features
            input_train_features , out_train_feature = train_df.drop(columns=[TARGET_COLUMN], axis=1) , train_df[TARGET_COLUMN]

            input_validatoin_features , out_validation_feature = validation_df.drop(columns=[TARGET_COLUMN], axis=1) , validation_df[TARGET_COLUMN]

            input_test_features , out_test_feature = test_df.drop(columns=[TARGET_COLUMN], axis=1) , test_df[TARGET_COLUMN]


            # Get the transformer object
            transformer = self.get_transformer()


            logging.info(f"fNumber of X_train:{input_train_features.shape}")
            logging.info(f"fNumber of X_test:{input_test_features.shape}")
            logging.info(f"fNumber of X_valid:{input_validatoin_features.shape}")

            # Perform the transformation on the train and test data
            logging.info("Applying transformer object")
            input_train_preprocessing_arr = transformer.fit_transform(input_train_features)

            input_validation_preprocessing_arr = transformer.fit_transform(input_validatoin_features)

            input_test_preprocessing_arr = transformer.transform(input_test_features)


            # Smotten object
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)


            # smotten on train
            logging.info("Applying SMOTEENN on Training dataset")
            input_train_final, output_train_final = smt.fit_resample(
                input_train_preprocessing_arr, out_train_feature
            )
            logging.info("Applied SMOTEENN on training dataset")


            #smotten on validation
            logging.info("Applying SMOTEENN on validation dataset")
            input_validation_final, output_validation_final = smt.fit_resample(
                input_validation_preprocessing_arr, out_validation_feature
            )
            logging.info("Applied SMOTEENN on validation dataset")


            # smotten on test
            logging.info("Applying SMOTEENN on testing dataset")
            input_test_final, output_test_final = smt.fit_resample(
                input_test_preprocessing_arr, out_test_feature
            )
            logging.info("Applied SMOTEENN on testing dataset")


            # combine the input arr and output feature
            train_arr = np.c_[input_train_final , np.array(output_train_final)]

            validation_arr = np.c_[input_validation_final, np.array(output_validation_final)]

            test_arr = np.c_[input_test_final , np.array(output_test_final)]


            #Save the the data array
            logging.info("saving data array into artifacts")
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,
                                        array=train_arr)
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_validation_path,
                                        array=validation_arr)
            
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,
                                        array=test_arr)

        
            # Save the pre-processing object
            logging.info("saving the transformer object")
            utils.save_object(file_path=self.data_transformation_config.pre_process_object_path,
                            obj=transformer)


            # save the data from data validation(helps in loading unique values in single prediction)
            data_obj = pd.read_csv(self.data_validation_artifacts.valid_train_path)
            utils.save_object(file_path=self.data_transformation_config.data_obj_path,
                              obj=data_obj)


            data_transformation_Artifact=artifacts_entity.DataTransformationArtifact(
                transform_train_path=self.data_transformation_config.transform_train_path,
                transform_validation_path=self.data_transformation_config.transform_validation_path,
                transform_test_path=self.data_transformation_config.transform_test_path,
                pre_process_object_path = self.data_transformation_config.pre_process_object_path)

            logging.info("returning data transformatin artifact")
            return data_transformation_Artifact

        except Exception as e:
            raise CustomException(e, sys)