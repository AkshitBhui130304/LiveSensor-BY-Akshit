import sys
from sensor.exception import SensorException


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