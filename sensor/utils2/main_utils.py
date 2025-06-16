import yaml
import pandas as pd
import dill
import os
import numpy as np
import pandas as pd
import sys
from sensor.exception import SensorException
from sensor.logger import logging

def read_yaml(file_path:str)->dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(f"Error reading YAML file at {file_path}: {e}", sys) from e



def write_yaml_file(file_path:str, content:object, replace:bool=False)->None:
    try:
        if replace:
            if( os.path.exists(file_path)):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)       
    except Exception as e:
        raise SensorException(f"Error writing YAML file at {file_path}: {e}",sys) from e
    

def save_numpy_array_data(file_path:str, array:np.array)->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise SensorException(f"Error saving numpy array to {file_path}: {e}",sys) from e   
    

def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise SensorException(f"Error loading numpy array from {file_path}: {e}", sys) from e 


def save_object(file_path:str, obj:object)->None:
    try:
        logging.info(f"Saving object to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise SensorException(f"Error saving object to {file_path}: {e}", sys) from e



def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise SensorException(f"File {file_path} does not exist.")
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise SensorException(f"Error loading object from {file_path}: {e}", sys) from e
    

    



