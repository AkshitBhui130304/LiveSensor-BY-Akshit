# Temporary debug script
from sensor.constant.database import DATABASE_NAME
from sensor.data_access.sensor_data import MongoDBClient

client = MongoDBClient()
print(client.client[DATABASE_NAME]["sensor_data"].count_documents({}))
