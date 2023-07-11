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
    