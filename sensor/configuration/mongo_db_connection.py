from dotenv import load_dotenv
import pymongo
from sensor.constant.database import DATABASE_NAME
import certifi
import os 
import logging
from sensor.constant.env_variable import MONGODB_URL_KEY

ca = certifi.where()
load_dotenv()

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                logging.info(f"Retrieved MongoDB URL: {mongo_db_url}")

                if not mongo_db_url:
                    raise ValueError("MONGO_DB_URL is not set in the .env file.")

                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            logging.error(f"Error initializing MongoDB client: {e}")
            raise e  # Don't write just `raise` without an exception object here
