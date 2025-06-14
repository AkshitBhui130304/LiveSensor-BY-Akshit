import yaml
import pandas as pd
import dill
import os
import numpy as np
import pandas as pd
import sys
from sensor.exception import SensorException


def read_yaml(file_path:str)->dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise SensorException(f"Error reading YAML file at {file_path}: {e}")
    


def write_yaml_file(file_path:str, content:object, replace:bool=False)->None:
    try:
        if replace:
            if( os.path.exists(file_path)):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)       
    except Exception as e:
        raise SensorException(f"Error writing YAML file at {file_path}: {e}")