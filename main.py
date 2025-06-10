from sensor.exception import SensorException
from sensor.logger import logging
import sys
import os


if __name__ == "__main__":
    try:
        logging.info("Starting the application")
        raise Exception("This is a test exception")
        
    except Exception as e:
        #print(e)
        raise SensorException(e, sys)