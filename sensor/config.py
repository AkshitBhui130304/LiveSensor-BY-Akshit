from dataclasses import dataclass
import os
import pymongo

@dataclass

class EnvironmentVariable:
    MONGO_DB_URL: str = os.getenv("MONGO_DB_URL")


env_variable = EnvironmentVariable()  

mongo_client = pymongo.MongoClient(env_variable.MONGO_DB_URL,
                                   connectTimeoutMS=100000,
                                   serverSelectionTimeoutMS=100000,
                                   socketTimeoutMS=100000)

