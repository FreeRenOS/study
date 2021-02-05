from pymongo import MongoClient


# client = MongoClient()
# client = MongoClient('localhost', '27017')
client = MongoClient('mongodb://127.0.0.1:27017')
for db_name in client.list_database_names():
    print(db_name)
