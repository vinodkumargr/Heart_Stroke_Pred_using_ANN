from src.logger import logging
from src.exception import CustomException
from src import entity
from src.entity import artifacts_entity, config_entity
from src import utils
from src.constants.database import TARGET_COLUMN
import pandas as pd
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from src.utils import read_yaml_file, write_yaml_file, append_yaml_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt


class ModelTrainer:
    def __init__(self, data_transformation_artifacts:artifacts_entity.DataTransformationArtifact,
                    model_trainer_config:config_entity.ModelTrainerConfig):
        try:

            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise CustomException(e, sys)
        
        

    def train_the_model(self, train_data: np.array, test_data: np.array, validation_data: np.array):
        try:
            self.X_train, self.y_train = train_data[:, :-1], train_data[:, -1]
            self.X_valid, self.y_valid = validation_data[:, :-1], validation_data[:, -1]
            self.X_test, self.y_test = test_data[:, :-1], test_data[:, -1]

            self.model = Sequential()
            self.model.add(Dense(128, activation='tanh', input_dim=16, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(192, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(288, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.1))
            self.model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dense(354, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            self.model.add(BatchNormalization())
            self.model.add(Dense(1, activation='sigmoid'))

            # Compile the model
            self.model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy'])

            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Define ReduceLROnPlateau callback
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

            # Fit the model with early stopping and learning rate reduction
            self.model_history = self.model.fit(
                self.X_train, self.y_train,
                epochs=100,
                validation_data=(self.X_valid, self.y_valid),
                validation_split=0.3,
                callbacks=[early_stopping, reduce_lr]
            )

            return self.model

        except Exception as e:
            raise CustomException(e, sys)



    def metrices_report(self, model:keras.models.Sequential):
        try:

            y_pred = model.predict(self.X_test)
            self.y_pred_classes = (y_pred > 0.5).astype("int32")
            self.score = accuracy_score(self.y_test, self.y_pred_classes)

            logging.info(f"model accuracy score : {self.score}")

            # Convert classification report to DataFrame
            cr_dict = classification_report(self.y_test, self.y_pred_classes, output_dict=True)
            cr_df = pd.DataFrame(cr_dict).transpose()

            # dataframe of classification report
            cr_df = cr_df.to_dict()

            # Write the DataFrame to a YAML file
            append_yaml_file(file_path=entity.MODEL_CONFIG_FILE, content=("Model evaluation metrices ......."))
            append_yaml_file(file_path=entity.MODEL_CONFIG_FILE, content=cr_df)

            logging.info(f"the report is : {cr_df}")

            return cr_df

        except Exception as e:
            raise CustomException(e, sys)
        

    def plot_graph_of_model_loss(self, model:str):
        try:

            plt.plot(self.model_history.history['loss'], color='black', label='Training Loss')
            plt.plot(self.model_history.history['val_loss'], color='red', label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.savefig(self.model_trainer_config.loss_graph)          

        except Exception as e:
            raise CustomException(e, sys) 


    def plot_graph_of_model_accuracy(self, model:str):
        try:

            plt.plot(self.model_history.history['accuracy'], color='black', label='Training Accuracy')
            plt.plot(self.model_history.history['val_accuracy'], color='red', label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.savefig(self.model_trainer_config.accuracy_graph)

            plt.close()          

        except Exception as e:
            raise CustomException(e, sys)
        


    def plot_confusion_metric_graph(self):
        try:

            # Compute the confusion matrix
            cm = confusion_matrix(self.y_test, self.y_pred_classes)

            # Define class labels
            class_labels = np.unique(self.y_test)

            # Plot the confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(class_labels))
            plt.xticks(tick_marks, class_labels)
            plt.yticks(tick_marks, class_labels)
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')

            # Add labels to each cell
            thresh = cm.max() / 2.0
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.savefig(self.model_trainer_config.confusion_metric_graph)          

        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_model_trainer(self) -> artifacts_entity.ModelTrainerArtifact:
        try:

            logging.info("Model trainer started ...")

            train_data = self.data_transformation_artifacts.transform_train_path
            validation_data = self.data_transformation_artifacts.transform_validation_path
            test_data = self.data_transformation_artifacts.transform_test_path


            train_arr = utils.load_numpy_array_data(train_data)
            validation_arr = utils.load_numpy_array_data(validation_data)
            test_arr = utils.load_numpy_array_data(test_data)

            pre_processing_obj = self.data_transformation_artifacts.pre_process_object_path

            logging.info("splitting train, valid and test data into x_train, x_test and x_valid, y_valid and y_train, y_test ...")

            x_train, y_train = train_arr[:, :-1] , train_arr[:, -1]
            X_valid, y_valid = validation_arr[:, :-1] , validation_arr[:, -1]
            x_test, y_test = test_arr[:, :-1] , test_arr[:, -1]


            # model tunning
            logging.info("Started model training ")


            # training the model
            logging.info("Training the model ")
            final_model = self.train_the_model(train_data=train_arr, test_data=test_arr, validation_data=validation_arr)
            logging.info("model triainedl")


            # metrices report
            logging.info("writing report of metrices into model.yaml file")
            final_model_report = self.metrices_report(model=final_model)


            # plot graphs and store into folder:
            logging.info("plotting graphs")
            self.plot_graph_of_model_loss(model=final_model_report)
            self.plot_graph_of_model_accuracy(model=final_model_report)
            self.plot_confusion_metric_graph()

            logging.info("observing model performance whether is underfitted or overfitted...")
            if self.score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"model expected r2_score is : {self.model_trainer_config.expected_r2_score}, but got model r2_score is : {self.score}")
            else:
                logging.info("model is good, performing well...")         

            logging.info("save the model as a pickle file....")
            utils.save_object(file_path=self.model_trainer_config.model_path, 
                                obj=final_model)
            
            logging.info("saving pre_process object")
            utils.save_object(file_path=self.model_trainer_config.pre_process_obj, obj=pre_processing_obj)

            model_trainer_artifact = artifacts_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                pre_processing_obj=self.model_trainer_config.pre_process_obj,
                r2_score = self.score
            )

            logging.info("exiting model trainer")

            return model_trainer_artifact


        except Exception as e:
            raise CustomException(e, sys)