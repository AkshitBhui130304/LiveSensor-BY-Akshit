from fastapi.responses import JSONResponse
import pandas as pd
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.exception import SensorException
from sensor.logger import logging
import sys
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.pipeline.training_pipeline import TrainPipeline
from fastapi import FastAPI, Response
from sensor.constant.application import APP_HOST, APP_PORT
from uvicorn import run as app_run
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from sensor.utils2.main_utils import load_object

app = FastAPI()

# Allow CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect root to docs
@app.get("/", tags=["root"])
async def index():
    return RedirectResponse(url="/docs")


# Route for training the model
@app.get("/train", tags=["pipeline"])
async def train():
    try:
       training_pipeline = TrainPipeline()

       if training_pipeline.is_pipeline_running:
        return Response("Training pipeline is already running")

       training_pipeline.run_pipeline()
       return Response("Training pipeline started successfully")
    
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        return Response(f"Error in training pipeline: {e}", status_code=500)


def main():
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        return {"message": "Training pipeline completed successfully"}
    except Exception as e:
        raise SensorException(e, sys) from e


# Dummy prediction route for example
@app.get("/predict", tags=["inference"])
async def predict():
    try:
      df = pd.read_csv("preprocessed_input.csv")  # Replace with actual data loading logic
      Model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
      if not Model_resolver.is_model_exists():
          return Response("Model does not exist", status_code=404)
      model = Model_resolver.get_best_model_path()

      new_model = load_object(file_path=model)
      y_pred = new_model.predict(df)
      df["predicted"] = y_pred
      df['predicted'].replace(TargetValueMapping().reverse_mapping(), inplace=True)
      
      result = df[["predicted"]].to_dict(orient="records")
      return JSONResponse(content={"predictions": result})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return Response(f"Error in prediction: {e}", status_code=500)  

# Entry point to run the app
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
