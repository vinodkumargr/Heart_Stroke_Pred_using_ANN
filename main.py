from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.entity import artifacts_entity, config_entity


if __name__ == "__main__":


    # data_ingestion:

    training_pipeline_config = config_entity.TrainingPipelineConfig()
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

    #data_validation

