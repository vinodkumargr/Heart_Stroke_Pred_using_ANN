import configs, os


# for configs
configs = os.path.abspath("configs")
SCHEMA_YAML_FILE = os.path.join(configs, "schema.yaml")
MODEL_YAML_FILE = os.path.join(configs, "model.yaml")


# for data ingestion
INGESTION_TRAIN_DATA_FILE = "ingestion_train_data.csv"
INGESTION_TEST_DATA_FILE = "ingestion_test_data.csv"
INGESTION_VALIDATION_DATA_FILE = "ingestion_validation_data.csv"


# for data validation
VALID_TRAIN_DATA_FILE = "valid_train_data.csv"
VALID_TEST_DATA_FILE = "valid_test_data.csv"
VALID_VALIDATION_DATA_FILE = "valid_validation_data.csv"


# for data transformation
TRANSFORMATION_TRAIN_DATA_FILE = "transform_train_data.npy"
TRANSFORMATION_TEST_DATA_FILE = "transform_test_data.npy"
TRANSFORMATION_VALIDATION_DATA_FILE = "transform_validation_data.npy"
TRANSFORMER_OBJ_FILE = "transformer_obj.pkl"
DATA_OBJ_FILE = "data_obj.pkl"


# for model trainer
MODEL_FILE = "model.pkl"
EXPECTED_ACCURACY = 0.8
MODEL_CONFIG_FILE = MODEL_YAML_FILE