import sys,os
from sensor.exception import SensorException
from sensor.constant.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR
from sensor.utils2.main_utils import load_object, save_object

class TargetValueMapping:

    def __init__(self):
        self.neg:int = 0
        self.pos:int = 1


    def to_dict(self) -> dict:
        """
        Convert the target value mapping to a dictionary.
        """
        return self.__dict__
    
    def reverse_mapping(self):
        """
        Reverse the target value mapping.
        """
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    

class SensorModel:
    def __init__(self,preprocessor, model):
        """
        Initialize the SensorModel with a model and target value mapping.
        """
        try:  
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise SensorException(e, sys) from e
        

    def predict(self, X):
        """
        Predict the target values for the given input data.
        """
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise SensorException(e, sys) from e
        

class ModelResolver:
    def __init__(self, model_dir = SAVED_MODEL_DIR):
        """
        Initialize the ModelResolver with the directory containing the model.
        """
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SensorException(e, sys) from e
        

    def get_best_model_path(self) -> str:
        try:
            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_version = max(timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_version}",MODEL_FILE_NAME)
            return latest_model_path
        
        except Exception as e:
            raise SensorException(e, sys) from e
        


    def is_model_exists(self) -> bool:
        """
        Check if the model exists in the specified directory.
        """
        try:
            if not os.path.exists(self.model_dir):
                return False
            
            timestamps = list(map(int, os.listdir(self.model_dir)))
            if len(timestamps) == 0:
                return False
            
            
            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False
            return True
        
        except Exception as e:
            raise SensorException(e, sys) from e    
        