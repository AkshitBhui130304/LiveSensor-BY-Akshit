import sys

import dill
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.utils2.main_utils import save_numpy_array_data, load_numpy_array_data, read_yaml
from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.ml.model.estimator import TargetValueMapping
import os
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline




class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            #self._schema_config = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise SensorException(e, sys)
        

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(f"Error reading data from {file_path}: {e}", sys)    

    
    
    
    
    
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            #target_value_mapping = TargetValueMapping()
            preprocessing_pipeline = Pipeline(steps=[
                ("imputer", simple_imputer),
                ("robust_scaler", robust_scaler)
            ])
            
            return  preprocessing_pipeline
        except Exception as e:
            raise SensorException(e, sys)
        

    def save_object(self, file_path: str, obj: object) -> None:
        try:
            logging.info(f"Saving object to {file_path}")
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)
                logging.info(f"Object saved to {file_path}")
        except Exception as e:
            raise SensorException(e, sys)
        


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(
                file_path=self.data_validation_artifact.valid_test_file_path)

            preprocessor = self.get_data_transformer_object()

            input_feature_train_def = train_df.drop(columns=[TARGET_COLUMN], axis=1)

            target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping().to_dict())

            input_feature_test_def = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_def = test_df[TARGET_COLUMN]  
           
            # Fit the preprocessor on the training data
            preprocessor_object = preprocessor.fit(input_feature_train_def)
            logging.info("Preprocessor fitted on training data")
            
            # Transform both training and test data
            transformed_input_feature_train = preprocessor_object.transform(input_feature_train_def)
            transformed_input_feature_test = preprocessor_object.transform(input_feature_test_def)

            smt = SMOTETomek(sampling_strategy='minority')
            
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_feature_train, target_feature_train_df
            )


            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_feature_test, target_feature_test_def
            )




            logging.info("SMOTE applied on training and test data")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("Data transformation completed")

            # Save the transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save the preprocessor object
            self.save_object(file_path=self.data_transformation_config.preprocessed_object_file_path, obj=preprocessor_object)
            logging.info("Transformed data and preprocessor object saved successfully")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path

            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
        except Exception as e:
            raise SensorException(e, sys)