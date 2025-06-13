from sensor.exception import SensorException
from sensor.logger import logging
import sys
import os
from sensor.utils import dump_csv_file_tomongodb

from sensor.pipeline.training_pipeline import TrainPipeline



if __name__ == "__main__":

    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
    
    
    
    