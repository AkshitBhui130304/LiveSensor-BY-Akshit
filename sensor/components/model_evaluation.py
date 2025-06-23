import pandas as pd
from sensor.exception import SensorException
from sensor.entity.config_entity import ModelEvaluationConfig,ModelTrainerConfig, DataValidationConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact, DataValidationArtifact, ModelTrainerArtifact
from sensor.logger import logging

import os,sys
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.utils2.main_utils import read_yaml, write_yaml_file, load_object, save_object
from sensor.constant.training_pipeline import MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE, TARGET_COLUMN
from sensor.ml.model.estimator import ModelResolver, SensorModel, TargetValueMapping



class ModelEvaluation:

    def __init__(self, model_evaluator_config: ModelEvaluationConfig,
                 model_trainer_artifact: ModelTrainerArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 ):
        """
        Initialize the ModelEvaluation with the ModelResolver.
        """
        try:
            self.model_evaluator_config = model_evaluator_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_validation_artifact = data_validation_artifact
            
        except Exception as e:
            raise SensorException(e, sys) from e    
    
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)
            y_train = self.data_validation_artifact.valid_train_file_path
            y_test = self.data_validation_artifact.valid_test_file_path
            
            train_df = pd.read_csv(y_train)
            test_df = pd.read_csv(y_test)

            df = pd.concat([train_df, test_df], axis=0)
            y_true = df[TARGET_COLUMN]
            y_true = y_true.replace(TargetValueMapping().to_dict())
            df.drop(columns=[TARGET_COLUMN], axis=1, inplace=True)
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted  = True

            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_file_path= None,
                    trained_model_file_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None
                )
                write_yaml_file(self.model_evaluator_config.model_evaluation_report_file_path,
                model_evaluation_artifact.__dict__
              )
                
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            train_model_metric_artifact = get_classification_score(y_true=y_true, y_pred=y_trained_pred)
            latest_model_metric_artifact = get_classification_score(y_true=y_true, y_pred=y_latest_pred)

            improved_accuracy = latest_model_metric_artifact.f1_score - train_model_metric_artifact.f1_score

            if self.model_evaluator_config.changed_threshold_score < improved_accuracy:
                is_model_accepted = True
            else:
                is_model_accepted = False

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_file_path=latest_model_path if is_model_accepted else None,
                trained_model_file_path=train_model_file_path,
                train_model_metric_artifact=train_model_metric_artifact,
                best_model_metric_artifact=latest_model_metric_artifact if is_model_accepted else None
            )

            write_yaml_file(self.model_evaluator_config.model_evaluation_report_file_path,
                model_evaluation_artifact.__dict__
              )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise SensorException(e, sys) from e