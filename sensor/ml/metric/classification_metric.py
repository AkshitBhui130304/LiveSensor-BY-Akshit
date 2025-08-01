from sensor.entity.config_entity import TrainingPipelineConfig
from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
import sys,os

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classsification_metric =  ClassificationMetricArtifact(f1_score=model_f1_score,
                    precision_score=model_precision_score, 
                    recall_score=model_recall_score)
        return classsification_metric
    except Exception as e:
        raise SensorException(e,sys)