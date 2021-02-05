from pymongo import MongoClient

client = MongoClient()
db = client.gridfs

for x in db.fs.files.find():
    print(x)

for x in db.fs.chunks.find():
    print(x)

