from pymongo import MongoClient
from datetime import datetime

current_date_time = datetime.now()
print(current_date_time)
# client = MongoClient()
# client = MongoClient('localhost', '27017')
client = MongoClient('mongodb://127.0.0.1:27017')
for db_name in client.list_database_names():
    print(db_name)
