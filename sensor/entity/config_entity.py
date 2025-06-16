from datetime import datetime

from sensor.constant import training_pipeline as tp
import os

class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()

        formatted_timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")

        self.pipeline_name = tp.PIPELINE_NAME
        self.artifact_dir = os.path.join(tp.ARTIFACT_DIR, formatted_timestamp)
        self.timestamp: str = formatted_timestamp

        

class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, tp.DATA_INGESTION_FEATURE_STORE_DIR, tp.FILENAME)
        self.ingested_train_file_path = os.path.join(self.data_ingestion_dir, tp.DATA_INGESTION_INGESTED_DIR, tp.TRAIN_FILE_NAME)
        self.ingested_test_file_path = os.path.join(self.data_ingestion_dir, tp.DATA_INGESTION_INGESTED_DIR, tp.TEST_FILE_NAME)
        self.train_test_split_ratio = tp.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name:str = tp.DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir :str = os.path.join(self.data_validation_dir, tp.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir :str = os.path.join(self.data_validation_dir, tp.DATA_VALIDATION_INVALID_DIR)

        self.valid_train_file_path:str = os.path.join(self.valid_data_dir, tp.TRAIN_FILE_NAME)
        self.valid_test_file_path:str = os.path.join(self.valid_data_dir, tp.TEST_FILE_NAME)
        self.invalid_train_file_path:str = os.path.join(self.invalid_data_dir, tp.TRAIN_FILE_NAME)
        self.invalid_test_file_path:str = os.path.join(self.invalid_data_dir, tp.TEST_FILE_NAME)
        self.drift_report_file_path:str = os.path.join(self.data_validation_dir, tp.DATA_VALIDATION_DRIFT_REPORT_DIR, tp.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)



class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, tp.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_data_dir = os.path.join(self.data_transformation_dir, tp.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        self.transformed_object_dir = os.path.join(self.data_transformation_dir, tp.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)

        self.transformed_train_file_path = os.path.join(self.transformed_data_dir, tp.TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_test_file_path = os.path.join(self.transformed_data_dir, tp.TEST_FILE_NAME.replace("csv", "npy"))
        self.preprocessed_object_file_path = os.path.join(self.transformed_object_dir, tp.PREPROCESSING_OBJECT_FILE_NAME)




class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, tp.MODEL_TRAINER_DIR_NAME)
        self.trained_model_dir = os.path.join(self.model_trainer_dir, tp.MODEL_TRAINER_TRAINED_MODEL_DIR)
        self.trained_model_file_path = os.path.join(self.trained_model_dir, tp.MODEL_TRAINER_TRAINED_MODEL_FILE_NAME)
        self.expected_accuracy = tp.MODEL_TRAINER_EXPECTED_ACCURACY
        self.over_fitting_under_fitting_threshold = tp.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD        