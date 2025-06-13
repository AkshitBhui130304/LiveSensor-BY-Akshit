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


