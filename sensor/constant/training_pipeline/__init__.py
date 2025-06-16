import os




TARGET_COLUMN = "class"
PIPELINE_NAME = "sensor"
ARTIFACT_DIR = "artifact"
FILENAME = "sensor.csv"

SAVED_MODEL_DIR = os.path.join("saved_models")


TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"


"""
data ingestion related constant values

"""

DATA_INGESTION_COLLECTION_NAME: str = "sensor_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
data validation related constant values

"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""data transformation related constant values"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



"""model trainer related constant values"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05

"""model evaluator related constant values"""
MODEL_EVALUATOR_DIR_NAME: str = "model_evaluator"
MODEL_EVALUATOR_REPORT_FILE_NAME: str = "report.yaml"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02