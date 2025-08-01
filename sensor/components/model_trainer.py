from sensor.utils2.main_utils import load_numpy_array_data,save_object,load_object
from sensor.entity.config_entity import ModelTrainerConfig
from sensor.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import SensorModel
from sensor.ml.metric.classification_metric import get_classification_score
from xgboost import XGBClassifier
import os,sys


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
        except Exception as e:
            raise SensorException(e, sys) from e
        
    def perform_hyperparameter_tuning(self, x_train, y_train):...
    
    
    
    
    def train_model(self, x_train, y_train):
        try:
            model = XGBClassifier()
            model.fit(x_train, y_train)
            return model
        except Exception as e:
            raise SensorException(e, sys) from e  


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model training")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_array = load_numpy_array_data(file_path=train_file_path)
            test_array = load_numpy_array_data(file_path=test_file_path)

            x_train, y_train = train_array[:,:-1], train_array[:,-1]
            x_test, y_test = test_array[:,:-1], test_array[:,-1]

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            classification_train_metric = get_classification_score(y_train, y_train_pred)

            if classification_train_metric.f1_score < self.model_trainer_config.expected_accuracy:
                raise SensorException(f"Model accuracy {classification_train_metric} is less than expected {self.model_trainer_config.expected_accuracy}", sys)

            classification_test_metric = get_classification_score(y_test, y_test_pred)
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            if diff > self.model_trainer_config.over_fitting_under_fitting_threshold:
                raise SensorException(f"Model is overfitting or underfitting. Train F1 Score: {classification_train_metric.f1_score}, Test F1 Score: {classification_test_metric.f1_score}, Difference: {diff}", sys)

            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_file_path}")
            preprocessor = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            sensor_model = SensorModel(preprocessor=preprocessor, model=model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=sensor_model)

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          train_metric_artifact=classification_train_metric,
                                                          test_metric_artifact=classification_test_metric)

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys) from e      
