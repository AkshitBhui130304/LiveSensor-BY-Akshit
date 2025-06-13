import pandas as pd
import numpy as np
import logging
import json
from sensor.config import mongo_client

def dump_csv_file_tomongodb(file_path:str,database_name:str,
    collection_name:str)->None:

    try:
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        json_records = list(json.loads(df.T.to_json()).values())

        mongo_client[database_name][collection_name].insert_many(json_records)


        print(f"Rows and columns: {df.shape}")
        print(f"Missing values: {df.isnull().sum().values.sum()}")

    except Exception as e:
        print(e)