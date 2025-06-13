import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import SensorException


class SensorData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
   
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise SensorException(e, sys)


    def save_csv_file(self,file_path ,collection_name: str, database_name: Optional[str] = None):
        try:
            data_frame=pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise SensorException(e, sys)


    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
               collection = self.mongo_client.database[collection_name]
               print(f"[DEBUG] Using default DB: {self.mongo_client.database.name}, Collection: {collection_name}")
            else:
                collection = self.mongo_client.client[database_name][collection_name]
                print(f"[DEBUG] Using DB: {database_name}, Collection: {collection_name}")

            data = list(collection.find())
            print(f"[DEBUG] Number of documents fetched: {len(data)}")

            df = pd.DataFrame(data)

            if "_id" in df.columns:
               df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            print(f"[DEBUG] Final DataFrame shape: {df.shape}")
            print(df.head())  # Optional: peek at actual data

            return df

        except Exception as e:
          raise SensorException(e, sys)

