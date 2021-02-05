import pymongo
import dns
client = pymongo.MongoClient("mongodb+srv://songshwan:1234@song-cluster.0nyqb.mongodb.net/song-db?retryWrites=true&w=majority")
for name in client.list_database_names():
    print(name)