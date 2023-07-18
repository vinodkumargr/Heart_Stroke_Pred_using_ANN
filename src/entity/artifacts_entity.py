from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_data_path:str
    test_data_path:str
    validation_data_path:str


@dataclass
class DataValidationArtifact:
    valid_train_path:str
    valid_test_path:str
    valid_validation_path:str
    

@dataclass
class DataTransformationArtifact:
    transform_train_path:str
    transform_validation_path:str
    transform_test_path:str
    pre_process_object_path:str


@dataclass
class ModelTrainerArtifact:
    model_path:str
    pre_processing_obj:str
    r2_score:float

@dataclass
class ModelEvaluationArtifact:
    model_accepted:bool
    improved_accuracy:float


@dataclass
class ModelPusherArtifact:
    pusher_model_dir:str
    saved_model_dir:str
