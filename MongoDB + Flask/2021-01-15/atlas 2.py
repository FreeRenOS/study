from pymongo import MongoClient
# 아래 atlas URI 서식을 위해 dnspython 라이브러리를 설치해야 함
import dns

client = MongoClient("mongodb+srv://songshwan:1234@song-cluster.0nyqb.mongodb.net/song-db?retryWrites=true&w=majority")
for name in client.list_database_names():
    print(name)
