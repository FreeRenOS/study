from pymongo import MongoClient
from datetime import datetime
# 아래 atlas URI 서식을 위해 dnspython 라이브러리를 설치해야 함
import dns

client = MongoClient("mongodb+srv://songshwan:1234@song-cluster.0nyqb.mongodb.net/"
                     "song-db?retryWrites=true&w=majority")
# for name in client.list_database_names():
#    print(name)
e= datetime.now().strftime('%Y--%d %m%H:%M:%S')
# datetim
datetime = datetime.now()

year = datetime.year
month = datetime.month
day = datetime.day

hour = datetime.hour
minute = datetime.minute
second = datetime.second

datetime = f'{year}-{month}-{day} {hour}:{minute}:{second}'

collection = client.test.books
booksData = [
    {
        "id": "01",
        "language": "Java",
        "edition": "third",
        "author": "Herbert Schildt",
        'published_date': datetime
    },

    {
        "id": "07",
        "language": "C++",
        "edition": "second",
        "author": "E.Balagurusamy",
        'published_date': datetime
    }
]

collection.insert_many(booksData)