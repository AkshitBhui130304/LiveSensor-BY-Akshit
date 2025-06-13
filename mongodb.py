from pymongo import MongoClient

uri = "mongodb+srv://Akshitbhui:Ak%40130304@cluster0.zwmbkyr.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("✅ Connected successfully.")
except Exception as e:
    print("❌ Connection error:", e)


db = client["AkshitBhui"]
collection = db["test"]

data = {
    "name": "Akshit Bhui",
    "age": 20,
    "city": "Kolkata"
}
collection.insert_one(data)
